"""Voxel rasterisation of sEEG electrodes into a head tissue label image.

Used by the CONFORMING embed method (``embed.build_conforming_head``): resolve a head
tissue label image, crop to the head, upsample to the target voxel size, and paint the
electrode materials (contact=13, shaft=14, glial sheath=15, fibrous sheath=16) directly into it -- replacing the
tissue they occupy, with NO artificial background block. The whole head is then remeshed
with CGAL so the electrode surfaces are honoured.

Pure numpy / scipy / nibabel (no SimNIBS mesh objects), so it stays a leaf below the
meshing layer.
"""

from __future__ import annotations

import os
import logging

import numpy as np

from ._params import (
    SEEG_CONTACT,
    SEEG_SHAFT,
    GLIAL_SHEATH,
    FIBROUS_SHEATH,
    DISPLACE_TAGS,
    SHEATH_TAG_BY_HOST,
)
from .geometry import perp_distance_and_axial

logger = logging.getLogger("simnibs.seeg")


# --------------------------------------------------------------------------- #
# reference resolution / crop / resample
# --------------------------------------------------------------------------- #
def resolve_reference(reference) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(label_image, affine)`` from a variety of inputs.

    ``reference`` may be an m2m folder (uses ``final_tissues.nii.gz``), a path to a
    label .nii.gz, or a ``(label_array, affine)`` tuple.
    """
    if isinstance(reference, tuple) and len(reference) == 2:
        lab, aff = reference
        return np.asarray(lab), np.asarray(aff, dtype=float)

    path = os.fspath(reference)
    if os.path.isdir(path):
        cand = os.path.join(path, "final_tissues.nii.gz")
        if not os.path.exists(cand):
            raise FileNotFoundError(
                f"no final_tissues.nii.gz in m2m folder {path!r}"
            )
        path = cand
    import nibabel as nib

    im = nib.load(path)
    lab = np.asarray(im.dataobj)
    if lab.ndim == 4:
        lab = lab[..., 0]
    return np.ascontiguousarray(lab), np.asarray(im.affine, dtype=float)


def crop_nonzero(label: np.ndarray, affine: np.ndarray, pad_mm: float = 6.0):
    """Crop to the nonzero (head) bounding box + pad; adjust the affine's origin."""
    nz = np.argwhere(label > 0)
    if nz.size == 0:
        return label, affine
    vox = np.linalg.norm(affine[:3, :3], axis=0)
    pad = np.ceil(pad_mm / vox).astype(int)
    lo = np.maximum(nz.min(0) - pad, 0)
    hi = np.minimum(nz.max(0) + pad + 1, label.shape)
    sub = label[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    aff = affine.copy()
    aff[:3, 3] = affine[:3, :3] @ lo + affine[:3, 3]
    return np.ascontiguousarray(sub), aff


def resample_iso(label: np.ndarray, affine: np.ndarray, vox_mm: float):
    """Nearest-neighbour resample to an isotropic ``vox_mm`` grid (labels preserved)."""
    from scipy.ndimage import zoom

    cur_vox = np.linalg.norm(affine[:3, :3], axis=0)
    factor = cur_vox / float(vox_mm)
    if np.allclose(factor, 1.0):
        return label, affine
    out = zoom(label, factor, order=0, mode="nearest")
    aff = affine.copy()
    # scipy.ndimage.zoom (grid_mode=False) aligns ENDPOINTS: out[0]=in[0], out[-1]=in[-1].
    # So the realized per-axis spacing is (n_in-1)/(n_out-1), not exactly 1/factor -- use the
    # realized spacing so the upsampled grid stays aligned to the world-mm leads.
    for i in range(3):
        denom = out.shape[i] - 1
        realized = (label.shape[i] - 1) / denom if denom > 0 else 1.0
        aff[:3, i] = affine[:3, i] * realized
    aff[:3, 3] = affine[:3, 3]   # voxel (0,0,0) world position unchanged (endpoint aligned)
    return np.ascontiguousarray(out.astype(label.dtype)), aff


# --------------------------------------------------------------------------- #
# electrode painting
# --------------------------------------------------------------------------- #
def _voxel_centres_world(affine, org, shape):
    gi = np.arange(org[0], org[0] + shape[0])
    gj = np.arange(org[1], org[1] + shape[1])
    gk = np.arange(org[2], org[2] + shape[2])
    I, J, K = np.meshgrid(gi, gj, gk, indexing="ij")
    V = np.stack([I.ravel(), J.ravel(), K.ravel()], axis=1).astype(float)
    return (affine[:3, :3] @ V.T + affine[:3, 3, None]).T


def _seg_voxel_bbox(A, B, r, inv, shape, pad=2):
    # all 8 corners of the padded cube around each endpoint -- for a rotated affine the
    # world AABB does NOT map to the voxel AABB, so 4 corners could undershoot.
    import itertools

    offs = np.array(list(itertools.product([r, -r], repeat=3)))
    corners = np.vstack([A + offs, B + offs])
    vc = (inv[:3, :3] @ corners.T + inv[:3, 3, None]).T
    lo = np.clip(np.floor(vc.min(0)).astype(int) - pad, 0, np.array(shape) - 1)
    hi = np.clip(np.ceil(vc.max(0)).astype(int) + pad + 1, 1, np.array(shape))
    return lo, hi


def paint_leads_into_label(
    label: np.ndarray,
    affine: np.ndarray,
    leads,
    catalog,
    sheath_mm: float,
    *,
    displace_tags: tuple[int, ...] | None = None,
    sheath_tag_map: dict[int, int] | None = None,
    entry_extend_mm: float = 4.0,
) -> tuple[np.ndarray, dict[int, int]]:
    """Paint contact (13), shaft (14) and the **segmented sheath** (15/16) into ``label``.

    The electrode is modelled over its **entire in-head length, skin -> tip**, and is wrapped
    in a **host-keyed** peri-electrode sheath: each sheath voxel is tagged by the tissue it
    grows in, so the glial scar (brain) and the fibrous tract (bone/scalp) carry their own
    conductivities, and compartments that host no reactive tissue (CSF) are left bare.

    * Body (contact discs + insulating shaft) replaces any tissue in ``displace_tags``
      (default 1..12) along the whole entry->tip trajectory. The entry is extended
      ``entry_extend_mm`` outward so the shaft reliably reaches the outer skin even if the
      supplied entry sits a little inside it (over-extension into air paints nothing).
    * Sheath is grown by morphological dilation of the body (>=1 voxel). Each shell voxel is
      mapped through ``sheath_tag_map`` (host tissue -> sheath tag): brain WM/GM -> GLIAL_SHEATH
      (15), bone/scalp/soft tissue -> FIBROUS_SHEATH (16); a host absent from the map (CSF,
      blood, ...) gets NO sheath (bare shaft there). Because it dilates the body, every
      face-neighbour of the body whose host is IN the map becomes sheath -- so the body is
      provably never face-adjacent to bare GM/WM (both mapped). Default map =
      ``_params.SHEATH_TAG_BY_HOST``. Priority sheath < shaft < contact.

    Returns ``(painted_label, voxel_counts)`` (counts keyed by 13/14/15/16).
    """
    from scipy.ndimage import binary_dilation

    out = label.copy()
    inv = np.linalg.inv(affine)
    voxel = float(np.mean(np.linalg.norm(affine[:3, :3], axis=0)))
    body_tags = np.asarray(DISPLACE_TAGS if displace_tags is None else displace_tags)
    if sheath_tag_map is None:
        sheath_tag_map = SHEATH_TAG_BY_HOST
    # host-tissue -> sheath-tag lookup table (index by host label; 0 = no sheath there). Span
    # through FIBROUS_SHEATH so already-painted sEEG voxels (13-16, all mapped to 0) from an
    # earlier lead are never re-tagged when they fall inside a later lead's shell.
    lut_n = int(max(list(sheath_tag_map) + [FIBROUS_SHEATH, int(out.max()), 0])) + 1
    host_lut = np.zeros(lut_n, dtype=out.dtype)
    for host, tag in sheath_tag_map.items():
        if 0 <= host < lut_n:
            host_lut[host] = tag
    counts = {SEEG_CONTACT: 0, SEEG_SHAFT: 0, GLIAL_SHEATH: 0, FIBROUS_SHEATH: 0}

    # sheath shell thickness in whole voxels, at TRUE thickness -- NOT forced to >=1. A sheath
    # thinner than ~half a voxel rounds to 0 and is not painted (the caller warns); it is never
    # widened to a voxel, because there is no conductivity scaling to compensate -- resolve a
    # thin sheath with a finer label voxel instead.
    n_sheath = int(round(max(0.0, sheath_mm) / voxel)) if sheath_mm > 0 else 0

    for lead in leads:
        spec = catalog[lead.part_number]
        A, B = lead.A, lead.B
        u = lead.axis_unit()
        # extend the entry outward so the rod reaches the outer skin (masked to tissue, so
        # any over-extension into air paints nothing). Contacts stay anchored to the real
        # tip, so this only lengthens the proximal insulating shaft.
        A_ext = A - u * float(entry_extend_mm)
        shift = float(entry_extend_mm)                      # |A_ext - A| along u
        # CONSTANT outer body radius: contacts (metal discs) and the insulating body share
        # the same Ø, differing only AXIALLY (by tag) -> smooth constant-diameter rod and a
        # constant-thickness sheath annulus (Karimi 2025; C1).
        r_body = spec.body_radius_mm
        r_sh = r_body + max(0.0, sheath_mm)
        s_c = (lead.contact_centers(spec) - A) @ u          # contact axial from real entry
        hlen = 0.5 * spec.contact_len_mm

        # bbox must also contain the full dilated shell
        r_pad = r_sh + (n_sheath + 2) * voxel
        lo, hi = _seg_voxel_bbox(A_ext, B, r_pad, inv, out.shape)
        if np.any(hi <= lo):
            continue
        shape = hi - lo
        P = _voxel_centres_world(affine, lo, shape)
        dist, axial = perp_distance_and_axial(P, A_ext, B)  # axial measured from A_ext
        axial_realA = axial - shift                         # -> axial from the real entry
        gap = (np.min(np.abs(axial_realA[:, None] - s_c[None, :]), axis=1)
               if s_c.size else np.full(axial.shape, np.inf))
        near = gap <= hlen

        cur = out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].reshape(-1)
        body_ok = np.isin(cur, body_tags)
        # within the rod: metal disc at a contact, insulating body elsewhere
        metal = body_ok & near & (dist <= r_body)
        shaft = body_ok & (dist <= r_body) & ~metal
        body_mask = (metal | shaft).reshape(shape)

        # segmented sheath tube: dilate the body, then tag each shell voxel by the tissue it
        # sits in (host_lut). Voxels whose host maps to a tag become sheath; hosts absent from
        # the map (CSF, blood, ...) get 0 -> stay bare. Because it dilates the body, every
        # face-neighbour of the body whose host is mapped is sheath -> no bare GM/WM (both mapped).
        if n_sheath > 0 and body_mask.any():
            shell = binary_dilation(body_mask, iterations=n_sheath).reshape(-1)
            host_idx = np.clip(cur, 0, host_lut.size - 1)
            sheath_tag_of = host_lut[host_idx]                 # per-voxel target sheath tag (0=none)
            sheath = shell & ~metal & ~shaft & (sheath_tag_of > 0)
        else:
            sheath = np.zeros_like(metal)
            sheath_tag_of = np.zeros_like(cur)

        # apply priority: sheath, then shaft, then metal (metal wins)
        cur[sheath] = sheath_tag_of[sheath]
        cur[shaft] = SEEG_SHAFT
        cur[metal] = SEEG_CONTACT
        out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = cur.reshape(shape)
        counts[SEEG_CONTACT] += int(metal.sum())
        counts[SEEG_SHAFT] += int(shaft.sum())
        counts[GLIAL_SHEATH] += int((sheath_tag_of[sheath] == GLIAL_SHEATH).sum())
        counts[FIBROUS_SHEATH] += int((sheath_tag_of[sheath] == FIBROUS_SHEATH).sum())

    return out, counts
