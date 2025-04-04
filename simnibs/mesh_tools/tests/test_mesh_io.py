import os
import copy
import tempfile

import numpy as np
import h5py
import pytest
from scipy.spatial import ConvexHull

from ... import SIMNIBSDIR
from .. import mesh_io
from ...utils import itk_mesh_io


@pytest.fixture(scope='module')
def sphere3_msh():
    fn = os.path.join(
            SIMNIBSDIR, '_internal_resources', 'testing_files', 'sphere3.msh')
    return mesh_io.read_msh(fn)


@pytest.fixture(scope='module')
def sphere3_baricenters(sphere3_msh):
    baricenters = np.zeros((sphere3_msh.elm.nr, 3), dtype=float)
    th_indexes = np.where(sphere3_msh.elm.elm_type == 4)[0]
    baricenters[th_indexes] = np.average(
        sphere3_msh.nodes.node_coord[
            sphere3_msh.elm.node_number_list[th_indexes, :4] - 1], axis=1)

    tr_indexes = np.where(sphere3_msh.elm.elm_type == 2)[0]
    baricenters[tr_indexes] = np.average(
        sphere3_msh.nodes.node_coord[
            sphere3_msh.elm.node_number_list[tr_indexes, :3] - 1], axis=1)
    return baricenters

@pytest.fixture
def atlas_itk_msh():
    fn = os.path.join(SIMNIBSDIR, '_internal_resources', 'testing_files', 'cube_atlas', 'atlas.txt.gz')
    return itk_mesh_io.itk_to_msh(fn)

class TestGetitem:
    def test_getitem_scalar(self):
        array = np.arange(1, 10, dtype=int)
        g = mesh_io._GetitemTester(array)
        assert g[1] == array[0]
        assert g[2] == array[1]
        assert g[-1] == array[-1]

    def test_getitem_slice(self):
        array = np.arange(1, 13, dtype=int).reshape(-1, 2)
        g = mesh_io._GetitemTester(array)
        assert np.all(g[1:10:2] == array[0:9:2])
        assert np.all(g[:] == array)
        assert np.all(g[1:] == array[0:])
        assert np.all(g[:5] == array[:4])
        assert np.all(g[::2] == array[::2])
        with pytest.raises(IndexError):
            g[::-1]
        with pytest.raises(IndexError):
            g[0::]
        with pytest.raises(IndexError):
            g[-1::]
        with pytest.raises(IndexError):
            g[::-1]

    def test_indexes(self):
        array = np.arange(1, 13, dtype=int).reshape(-1, 2)
        g = mesh_io._GetitemTester(array)
        assert np.all(g[[1, 3, 4]] == array[[0, 2, 3]])
        assert np.all(g[np.array([1, 3, 4])] == array[np.array([0, 2, 3])])
        with pytest.raises(IndexError):
            g[[0]]
        assert np.all(g[[1, 5], [0, 1]] == array[[0, 4], [0, 1]])
        idx = np.array([[1, 3],
                        [5, 2]])
        assert np.all(g[idx] == array[idx-1])

    def test_two_slices(self):
        array = np.arange(1, 101, dtype=int).reshape(-1, 5)
        g = mesh_io._GetitemTester(array)
        assert np.all(g[1:10:2, :3] == array[0:9:2, :3])

    def test_bool(self):
        array = np.arange(1, 101, dtype=int).reshape(-1, 5)
        g = mesh_io._GetitemTester(array)
        assert np.all(g[array[:,0]>20] == array[array[:,0]>20])
        assert np.all(g[array>20] == array[array>20])

    def test_ellipsis(self):
        array = np.arange(1, 101, dtype=int).reshape(-1, 5)
        g = mesh_io._GetitemTester(array)
        assert np.all(g[...] == array[...])
        assert np.all(g[..., 1:3] == array[..., 1:3])

    def test_None(self):
        array = np.arange(1, 101, dtype=int).reshape(-1, 5)
        g = mesh_io._GetitemTester(array)
        assert np.all(g[None] == array[None])
        assert np.all(g[np.newaxis, 1:3] == array[np.newaxis, 1:3])
        assert np.all(g[:, 1:3, None] == array[:, 1:3, None])

    def test_mixed(self):
        array = np.arange(1, 101, dtype=int).reshape(-1, 5)
        g = mesh_io._GetitemTester(array)
        assert np.all(g[[1,3,5], 1:3] == array[[0,2,4], 1:3])
        assert np.all(g[array[:,0]>20, 1:3] == array[array[:,0]>20, 1:3])

    def test_advanced(self):
        array = np.arange(1, 101, dtype=int).reshape(-1, 5)
        rows = np.array([0, 5])
        cols = np.array([0, 3])
        g = mesh_io._GetitemTester(array)
        assert np.all(g[rows+1, cols] == array[rows, cols])
        assert np.all(g[np.ix_(rows+1, cols)] == array[np.ix_(rows, cols)])


class TestGmshRead:

    def test_node_nr(self, sphere3_msh):
        assert sphere3_msh.nodes.nr == 4556

    def test_node_coord(self, sphere3_msh):
        np.testing.assert_array_almost_equal(
            sphere3_msh.nodes.node_coord[-1, :],
            np.array([-27.4983367, 72.72180083, 9.66883006]))

    def test_elm_nr(self, sphere3_msh):
        assert sphere3_msh.elm.nr == 30530

    def test_elm_tag1(self, sphere3_msh):
        assert np.unique(sphere3_msh.elm.tag1).tolist() == [
            3, 4, 5, 1003, 1004, 1005]

    def test_elm_node_number_list(self, sphere3_msh):
        np.testing.assert_array_equal(sphere3_msh.elm.node_number_list[-1, :],
                                      np.array([31, 4149, 4272, 1118]))


class TestNodes:

    def test_node_init_nrs(self):
        node_coord = np.array(((3, 2, 4), (5, 3, 3)))
        nodes = mesh_io.Nodes(node_coord)
        assert nodes.nr == 2

    def test_node_init_node_nr(self):
        node_coord = np.array(((3.0, 2.0, 4), (5, 3, 3)))
        nodes = mesh_io.Nodes(node_coord)
        np.testing.assert_array_almost_equal(
            nodes.node_number, np.array((1, 2)))

    def test_node_find_closest_node(self):
        node_coord = np.array(((3, 2, 4), (5, 3, 3), (1, 2, 0)))
        nodes = mesh_io.Nodes(node_coord)
        n = nodes.find_closest_node(
            np.array(((3, 1.9, 4), (5.2, 2.2, 3), (1, 2, 1)), dtype=float))

        np.testing.assert_array_equal(node_coord, n)

    def test_node_find_closest_testing(self):
        node_coord = np.array(((3, 2, 4), (5, 3, 3), (1, 2, 0)))
        nodes = mesh_io.Nodes(node_coord)
        _, idx = nodes.find_closest_node(np.array(
            ((3, 1.9, 4), (5.2, 2.2, 3), (1, 2, 1)), dtype=float),
            return_index=True)

        assert all(idx == [1, 2, 3])

    def test_getitem_simple_index(self):
        nodes = mesh_io.Nodes()
        nodes.node_coord = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert np.all(nodes[1] == [1, 0, 0])

    def test_getitem_list_index(self):
        nodes = mesh_io.Nodes()
        nodes.node_coord = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert np.all(nodes[[1, 2]] == [[1, 0, 0], [0, 1, 0]])
        assert np.all(nodes[np.array([1, 2])] == [[1, 0, 0], [0, 1, 0]])

    def test_getitem_slice(self):
        nodes = mesh_io.Nodes()
        nodes.node_coord = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert np.all(nodes[1:] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_getitem_raise(self):
        nodes = mesh_io.Nodes()
        nodes.node_coord = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        with pytest.raises(IndexError):
            nodes[0]

class TestElements:
    def test_elm_init_nr(self):
        triangles = np.array(((1, 3, 2), (4, 1, 2)))
        tetrahedra = np.array(((1, 3, 2, 4), (4, 1, 3, 7)))
        elm = mesh_io.Elements(triangles, tetrahedra)
        assert elm.nr == 4
        assert np.all(elm[:] == np.array(
            [[1, 3, 2, -1], [4, 1, 2, -1],
             [1, 3, 2, 4], [4, 1, 3, 7]]))

    def test_elm_init_node_number_list(self):
        triangles = np.array(((1, 3, 2), (4, 1, 2)))
        tetrahedra = np.array(((1, 3, 2, 4), (4, 1, 3, 7)))
        elm = mesh_io.Elements(triangles, tetrahedra)
        np.testing.assert_array_almost_equal(
            elm.node_number_list,
            np.array(((1, 3, 2, -1), (4, 1, 2, -1), (1, 3, 2, 4), (4, 1, 3, 7))))

    def test_elements_find_neighbouring_nodes(self):
        elm = mesh_io.Elements()
        elm.node_number_list = np.array(
            [[1, 2, 3, 4], [3, 2, 5, 7], [1, 8, 2, 6]])
        assert set(elm.find_neighbouring_nodes(1)) == set([2, 3, 4, 8, 6])

    def test_elements_find_neighbouring_elements(self):
        elm = mesh_io.Elements()
        elm.node_number_list = np.array(
            [[1, 2, 3, 4], [3, 2, 5, 7], [1, 8, 2, 6]])
        assert set(elm.find_all_elements_with_node(1)) == set([1, 3])

    def test_elements_find_neighbouring_elements_many(self):
        elm = mesh_io.Elements()
        elm.node_number_list = np.array(
            [[1, 2, 3, 4], [3, 2, 5, 7], [1, 8, 2, 6]])
        assert set(elm.find_all_elements_with_node([3, 7])) == set([1, 2])

    def test_elements_getitem(self):
        elm = mesh_io.Elements()
        elm.node_number_list = np.array(
            [[1, 2, 3, 4], [3, 2, 5, 7], [1, 8, 2, 6]])
        assert np.all(elm[[2, 3]] == np.array([[3, 2, 5, 7], [1, 8, 2, 6]]))

    def test_get_faces(self, sphere3_msh):
        elm = sphere3_msh.elm
        faces, th_faces, adjacency_list = elm.get_faces()
        th = elm[elm.tetrahedra]
        faces_th = th[:, [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]]
        assert np.all(faces_th[0] == faces[th_faces[0]])
        assert np.sum(np.isin(th[adjacency_list[0, 0]], th[adjacency_list[0, 1]])) == 3
        assert np.any(np.isin(th_faces[0], th_faces[adjacency_list[th_faces[0, 0], 1]]))

        # Here I don't care about order because it might be flipped
        assert np.all(np.unique(faces_th[-1]) == np.unique(faces[th_faces[-1]]))
        assert np.sum(np.isin(th[adjacency_list[-1, 0]], th[adjacency_list[-1, 1]])) == 3
        assert np.any(np.isin(th_faces[-1], th_faces[adjacency_list[th_faces[-1, 0], 1]]))


    def test_get_outside_faces(self, sphere3_msh):
        elm = sphere3_msh.elm
        outside_faces = elm.get_outside_faces()
        nodes_in_face = sphere3_msh.nodes[outside_faces]
        assert np.allclose(np.linalg.norm(nodes_in_face, axis=2), 95)
        edge1 = nodes_in_face[:, 1, :] - nodes_in_face[:, 0, :]
        edge2 = nodes_in_face[:, 2, :] - nodes_in_face[:, 0, :]
        n = np.cross(edge1, edge2)
        n /= np.linalg.norm(n, axis=1)[:, None]
        bar = np.average(nodes_in_face, axis=1)
        bar /= np.linalg.norm(bar, axis=1)[:, None]
        assert np.allclose(n, bar, atol=1e-1)

    def test_nodes_in_tag(self, sphere3_msh):
        n = sphere3_msh.elm.nodes_with_tag(5)
        r = np.linalg.norm(sphere3_msh.nodes[n], axis=1)
        assert np.all(r > 89.) and np.all(r < 96.)

        n = sphere3_msh.elm.nodes_with_tag([3, 5])
        r = np.linalg.norm(sphere3_msh.nodes[n], axis=1)
        assert np.all(((r > 89.)*(r < 96.)) + (r < 86.))

    def test_find_adjacent_tetrahedra_tr(self, sphere3_msh):
        adj_elements = sphere3_msh.elm.find_adjacent_tetrahedra()
        outer_surf = sphere3_msh.elm.tag1 == 1005
        assert np.all(adj_elements[outer_surf, 1:] == -1)
        assert np.all(sphere3_msh.elm.tag1[adj_elements[outer_surf, 0] - 1] == 5)
        inner = sphere3_msh.elm.tag1 == 1004
        assert np.all(np.isin(sphere3_msh.elm.tag1[adj_elements[inner, :1] - 1], [4, 5]))

    def test_find_adjacent_tetrahedra_tr_dangling(self, sphere3_msh):
        m = sphere3_msh.crop_mesh([3, 1005])
        adj_elements = m.elm.find_adjacent_tetrahedra()
        outer_surf = m.elm.tag1 == 1005
        assert np.all(adj_elements[outer_surf] == -1)

    def test_find_adjacent_tetrahedra_th(self, sphere3_msh):
        adj_elements = sphere3_msh.elm.find_adjacent_tetrahedra()
        inner_vol = sphere3_msh.elm.tag1 == 3
        assert np.all(np.unique(sphere3_msh.elm.tag1[adj_elements[inner_vol] - 1]) == [3, 4])
        mid = sphere3_msh.elm.tag1 == 4
        assert np.all(np.unique(sphere3_msh.elm.tag1[adj_elements[mid] - 1]) == [3, 4, 5])

    def test_connected_components_connected_mesh(self, sphere3_msh):
        cc = sphere3_msh.elm.connected_components()
        assert len(cc) == 1
        assert np.all(cc[0] == sphere3_msh.elm.elm_number)

    def test_connected_components_disconnected_th(self, sphere3_msh):
        elm_select = sphere3_msh.elm.elm_number[
            (sphere3_msh.elm.tag1 == 3) +
            (sphere3_msh.elm.tag1 == 5)
        ]
        cc = sphere3_msh.elm.connected_components(elm_select)
        assert len(cc) == 2
        assert np.all(cc[0] == sphere3_msh.elm.elm_number[(sphere3_msh.elm.tag1 == 3)])
        assert np.all(cc[1] == sphere3_msh.elm.elm_number[(sphere3_msh.elm.tag1 == 5)])

    def test_connected_components_disconnected_th_tr(self, sphere3_msh):
        elm_select = sphere3_msh.elm.elm_number[
            (sphere3_msh.elm.tag1 == 3) +
            (sphere3_msh.elm.tag1 == 1005)
        ]
        cc = sphere3_msh.elm.connected_components(elm_select)
        assert len(cc) == 2
        assert np.all(cc[0] == sphere3_msh.elm.elm_number[(sphere3_msh.elm.tag1 == 3)])
        assert np.all(cc[1] == sphere3_msh.elm.elm_number[(sphere3_msh.elm.tag1 == 1005)])

    def test_add_triangles(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        m.elm.add_triangles(m.elm[m.elm.tag1 == 1005, :3], 1006)
        assert m.elm.nr == sphere3_msh.elm.nr + np.sum(m.elm.tag1 == 1005)
        assert np.all(
            m.elm[m.elm.tag1 == 1006] ==
            sphere3_msh.elm[sphere3_msh.elm.tag1 == 1005]
        )

    def test_node_elm_adjacency(self, sphere3_msh):
        M = sphere3_msh.elm.node_elm_adjacency()
        for n in [1, 523, 3803, 1821, sphere3_msh.nodes.nr]:
            elm_with_node = sphere3_msh.elm.elm_number[
                np.any(sphere3_msh.elm.node_number_list == n, axis=1)
            ]
            assert np.all(M[n].data == elm_with_node)

    def test_add_triangles_tags(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        tr = m.elm[m.elm.tag1 == 1005, :3]
        tags = m.elm.tag1[m.elm.tag1 == 1005] + 1
        m.elm.add_triangles(tr, tags)
        assert m.elm.nr == sphere3_msh.elm.nr + np.sum(m.elm.tag1 == 1005)
        assert np.all(
            m.elm[m.elm.tag1 == 1006] ==
            sphere3_msh.elm[sphere3_msh.elm.tag1 == 1005]
        )



class TestMsh:
    def test_compact_ordering_elm(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        m.elm.node_number_list += 10
        m.compact_ordering(m.nodes.node_number + 10)
        assert np.all(m.elm.node_number_list == sphere3_msh.elm.node_number_list)


    def test_crop_mesh(self, sphere3_msh):
        cropped_mesh = sphere3_msh.crop_mesh([1003, 5])
        nr_elements_tags = np.sum(np.logical_or(cropped_mesh.elm.tag1 == 1003,
                                                cropped_mesh.elm.tag1 == 5))
        assert nr_elements_tags == cropped_mesh.elm.nr

    def test_crop_mesh_type(self, sphere3_msh):
        cropped_mesh = sphere3_msh.crop_mesh(elm_type=4)
        assert np.all(cropped_mesh.elm.elm_type == 4)

    def test_crop_mesh_nodes(self, sphere3_msh):
        target = range(1, 11)
        w_node = np.any(np.isin(sphere3_msh.elm.node_number_list,
                                target).reshape(-1,4), axis=1)
        neighbours = np.unique(sphere3_msh.elm.node_number_list[w_node].T)[1:]
        orig_coords = sphere3_msh.nodes[neighbours]
        cropped_mesh = sphere3_msh.crop_mesh(nodes=range(1, 11))
        assert np.allclose(orig_coords, cropped_mesh.nodes.node_coord)

    def test_crop_mesh_elements(self, sphere3_msh):
        target = [5, 104, 405, 1000]
        bar = sphere3_msh.elements_baricenters()[target]
        cropped_mesh = sphere3_msh.crop_mesh(elements=target)
        assert np.allclose(bar, cropped_mesh.elements_baricenters().value)
        assert np.allclose(sphere3_msh.elm.tag1[np.array(target) - 1],
                           cropped_mesh.elm.tag1)

    def test_join_mesh(self, sphere3_msh):
        m1 = sphere3_msh.crop_mesh([3, 1003])
        v1 = m1.elements_volumes_and_areas().value.sum()
        m2 = sphere3_msh.crop_mesh([5, 1005])
        v2 = m2.elements_volumes_and_areas().value.sum()
        m = m1.join_mesh(m2)
        v = m.elements_volumes_and_areas().value.sum()
        assert m.elm.nr == m1.elm.nr + m2.elm.nr
        assert m.nodes.nr == m1.nodes.nr + m2.nodes.nr
        assert np.isclose(v, v1+v2)

    def test_join_mesh_data(self, sphere3_msh):
        m1 = sphere3_msh.crop_mesh([3, 1003])
        m1.nodedata = [mesh_io.NodeData(value=m1.nodes.node_coord)]
        m1.elmdata = [m1.elements_baricenters()]
        m2 = sphere3_msh.crop_mesh([5, 1005])
        m2.nodedata = [mesh_io.NodeData(value=m2.nodes.node_number)]
        m2.elmdata = [mesh_io.ElementData(value=m2.elm.tag1)]
        m = m1.join_mesh(m2)
        assert np.all(np.isclose(m.nodedata[0].value[:m1.nodes.nr],m1.nodes.node_coord))
        assert np.all(np.isnan(m.nodedata[0].value[m1.nodes.nr:]))

        v = m.elmdata[0].value
        ref = m1.elmdata[0].value
        assert np.all(np.isclose(v[np.isin(m.elm.tag1, [3, 1003])], ref))
        assert np.all(np.isnan(v[~np.isin(m.elm.tag1, [3, 1003])]))

        assert np.all(np.isclose(m.nodedata[1].value[m1.nodes.nr:],m2.nodes.node_number))
        assert np.all(np.isnan(m.nodedata[1].value[:m1.nodes.nr]))

        v = m.elmdata[1].value
        ref = m2.elmdata[0].value
        assert np.all(np.isclose(v[np.isin(m.elm.tag1, [5, 1005])], ref))
        assert np.all(np.isnan(v[~np.isin(m.elm.tag1, [5, 1005])]))


    def test_remove_elements(self, sphere3_msh):
        removed = sphere3_msh.remove_from_mesh(1003)
        assert np.all(removed.elm.tag1 != 1003)

    def test_remove_mesh_type(self, sphere3_msh):
        removed = sphere3_msh.remove_from_mesh(elm_type=4)
        assert np.all(removed.elm.elm_type == 2)

    def test_remove_mesh_nodes(self, sphere3_msh):
        removed = sphere3_msh.remove_from_mesh(
            nodes=np.unique(sphere3_msh.elm[sphere3_msh.elm.tag1==5])
        )

        assert len(np.setdiff1d([3, 1003, 4], removed.elm.tag1)) == 0

    def test_remove_mesh_elements(self, sphere3_msh):
        removed = sphere3_msh.remove_from_mesh(
            elements=sphere3_msh.elm.elm_number[sphere3_msh.elm.tag1==1005]
        )
        assert len(np.setdiff1d([3, 1003, 4, 1004, 5], removed.elm.tag1)) == 0


    def test_elements_baricenters(self, sphere3_msh):
        baricenters = sphere3_msh.elements_baricenters()

        b = np.zeros((sphere3_msh.elm.nr, 3), dtype=float)
        th_indexes = np.where(sphere3_msh.elm.elm_type == 4)[0]
        b[th_indexes] = np.average(
            sphere3_msh.nodes.node_coord[
                sphere3_msh.elm.node_number_list[th_indexes, :4] - 1], axis=1)

        tr_indexes = np.where(sphere3_msh.elm.elm_type == 2)[0]
        b[tr_indexes] = np.average(
            sphere3_msh.nodes.node_coord[
                sphere3_msh.elm.node_number_list[tr_indexes, :3] - 1], axis=1)

        np.testing.assert_almost_equal(baricenters.value, b)

    def test_areas(self, sphere3_msh):
        v = sphere3_msh.elements_volumes_and_areas()
        tot_area = np.sum(v[sphere3_msh.elm.triangles])

        np.testing.assert_allclose(tot_area, 4 * np.pi * (95**2 + 90**2 + 85**2), rtol=1e-1)

    def test_volume(self, sphere3_msh):
        v = sphere3_msh.elements_volumes_and_areas()
        tot_vol = np.sum(v[sphere3_msh.elm.tetrahedra])

        # high tolerance due to low resolution
        np.testing.assert_allclose(tot_vol, 4. / 3. * np.pi * 95**3, rtol=2e-1)

    def test_find_closest_element(self, sphere3_msh):
        b = sphere3_msh.elements_baricenters().value
        c = sphere3_msh.find_closest_element(b)

        np.testing.assert_allclose(b, c)

    def test_triangle_normals(self, sphere3_msh):
        n = sphere3_msh.triangle_normals()
        _, top = sphere3_msh.find_closest_element([95, 0, 0], return_index=True,
                                                   elements_of_interest=sphere3_msh.elm.triangles)

        # high tolerance due to low resolution
        assert np.allclose(n[top], [1., 0., 0.], rtol=1e-1, atol=1e-1)

    def test_triangle_normals_smooth(self, sphere3_msh):
        n = sphere3_msh.triangle_normals(smooth=1)
        _, top = sphere3_msh.find_closest_element(
            [95, 0, 0], return_index=True,
            elements_of_interest=sphere3_msh.elm.triangles)
        # high tolerance due to low resolution
        assert np.allclose(n[top], [1., 0., 0.], rtol=1e-1, atol=1e-1)


    def test_find_closest_element_index(self, sphere3_msh):
        b = sphere3_msh.elements_baricenters().value
        c, idx = sphere3_msh.find_closest_element(b, return_index=True)
        assert np.all(idx == sphere3_msh.elm.elm_number)

    def test_element_node_coords_all(self, sphere3_msh):
        tmp_node_coors = np.vstack((sphere3_msh.nodes.node_coord, [0, 0, 0]))

        node_coords = tmp_node_coors[sphere3_msh.elm.node_number_list - 1]

        np.testing.assert_equal(
            node_coords,
            sphere3_msh.elm_node_coords())

    def test_element_node_coords_tag(self, sphere3_msh):
        node_coords = sphere3_msh.elm_node_coords(tag=2)

        assert np.all(np.linalg.norm(node_coords, axis=2) < 86)

    def test_write_mesh_to_hdf5(self):
        m = mesh_io.Msh()
        m.fn = 'test'
        m.elm.tag1 = np.array([1, 2, 3])
        m.elm.node_number_list = np.array([4, 5, 6])
        m.elmdata.append(mesh_io.ElementData(np.arange(9).reshape(3, 3), 'E'))
        m.nodedata.append(mesh_io.ElementData(np.array([7, 8, 9]), 'V'))
        if os.path.isfile('tmp.hdf5'):
            os.remove('tmp.hdf5')
        m.write_hdf5('tmp.hdf5')
        with h5py.File('tmp.hdf5', 'r') as f:
            assert f.attrs['fn'] == 'test'
            elm = f['elm']
            assert set(elm.keys()) == set(m.elm.__dict__.keys())
            np.testing.assert_equal([1, 2, 3], elm['tag1'][:])
            nodes = f['nodes']
            assert set(nodes.keys()) == set(m.nodes.__dict__.keys())
            elmdata = f['elmdata']
            assert list(elmdata.keys()) == ['E']
            np.testing.assert_equal(np.array(elmdata['E']), np.arange(9).reshape(3, 3))
            nodedata = f['nodedata']
            assert list(nodedata.keys()) == ['V']
            np.testing.assert_equal(np.array(nodedata['V']), [7, 8, 9])

        os.remove('tmp.hdf5')

    def test_read_hdf5(self):
        if os.path.isfile('tmp.hdf5'):
            os.remove('tmp.hdf5')

        with h5py.File('tmp.hdf5', 'w') as f:
            f.attrs['fn'] = 'path/to/file.msh'
            elm = f.create_group('elm')
            e = mesh_io.Elements()
            e.tag1 = np.array([3, 2, 1])
            for key, value in e.__dict__.items():
                elm.create_dataset(key, data=value)
            n = mesh_io.Nodes()
            n.node_coord = np.array([0, 0, 0])
            nodes = f.create_group('nodes')
            for key, value in n.__dict__.items():
                nodes.create_dataset(key, data=value)
            elmdata = f.create_group('elmdata')
            elmdata.create_dataset('E', data=np.arange(9).reshape(3, 3))
            nodedata = f.create_group('nodedata')
            nodedata.create_dataset('V', data=np.array([7, 8, 9]))

        m = mesh_io.Msh.read_hdf5('tmp.hdf5')
        np.testing.assert_equal([3, 2, 1], m.elm.tag1)
        np.testing.assert_equal([0, 0, 0], m.nodes.node_coord)
        assert len(m.elmdata) == 1
        assert m.elmdata[0].field_name == 'E'
        np.testing.assert_equal(np.arange(9).reshape(3, 3), m.elmdata[0].value)
        assert m.nodedata[0].field_name == 'V'
        np.testing.assert_equal([7, 8, 9], m.nodedata[0].value)

        os.remove('tmp.hdf5')

    def test_read_and_write_hdf5(self, sphere3_msh):
        sphere3_msh.elmdata.append(mesh_io.ElementData(np.array([1, 2, 3, 4]), 'elm'))
        sphere3_msh.nodedata.append(mesh_io.NodeData(np.array([5, 6, 7, 8]), 'nd'))
        sphere3_msh.write_hdf5('tmp.hdf5', path='msh/')

        m = mesh_io.Msh.read_hdf5('tmp.hdf5', path='msh/')
        np.testing.assert_equal(sphere3_msh.elm.tag1, m.elm.tag1)
        np.testing.assert_equal(sphere3_msh.elm.tag2, m.elm.tag2)
        np.testing.assert_equal(sphere3_msh.elm.node_number_list, m.elm.node_number_list)
        np.testing.assert_equal(sphere3_msh.nodes.node_coord, m.nodes.node_coord)
        assert m.elmdata[0].field_name == 'elm'
        assert m.nodedata[0].field_name == 'nd'
        np.testing.assert_equal([1, 2, 3, 4], m.elmdata[0].value)
        np.testing.assert_equal([5, 6, 7, 8], m.nodedata[0].value)
        sphere3_msh.elmdata = []
        sphere3_msh.nodedata = []
        os.remove('tmp.hdf5')

    def test_quality_parameters(self):
        # define mesh with a single regular tetrahedron
        msh = mesh_io.Msh()
        # 2 regular tetrahedron of edge size 1
        msh.elm = mesh_io.Elements(
            tetrahedra=np.array(
                [[1, 2, 3, 4],
                 [1, 2, 5, 4]], dtype=int))
        msh.elm.elm_type = np.array([4, 4])
        msh.nodes = mesh_io.Nodes(np.array(
            [[1./3. * np.sqrt(3), 0, 0],
             [-1./6. * np.sqrt(3), 1./2., 0],
             [0, 0, 1./3. * np.sqrt(6)],
             [-1./6. * np.sqrt(3), -1./2., 0],
             [0, 0, 1./3. * np.sqrt(6)]], dtype=float))
        re_ratio, ir_cr_ratio = msh.tetrahedra_quality()
        # circumradius of a regular tetrahedron
        # http://mathworld.wolfram.com/RegularTetrahedron.html
        assert np.allclose(re_ratio[:], 1/4*np.sqrt(6))
        assert np.allclose(ir_cr_ratio[:], 1/3)

    def test_triangle_angles(self):
        msh = mesh_io.Msh()
        # 2 regular tetrahedron of edge size 1
        msh.elm = mesh_io.Elements(
            triangles=np.array(
                [[1, 2, 3],
                 [1, 2, 3],
                 [1, 4, 2],
                 [1, 4, 2]], dtype=int))
        msh.nodes = mesh_io.Nodes(np.array(
            [[0, 0, 0],
             [1, 0, 0],
             [0, np.tan(np.pi/6), 0],
             [0, 0, np.tan(np.pi/6)]], dtype=float))
        angles = msh.triangle_angles()
        assert np.allclose(angles[:3], [90, 30, 60])
        assert np.allclose(angles[3:], [90, 60, 30])

    def test_curvature(self, sphere3_msh):
        # https://computergraphics.stackexchange.com/a/1721
        surf_1005 = np.unique(sphere3_msh.elm[sphere3_msh.elm.tag1 == 1005, :3])
        surf_1004 = np.unique(sphere3_msh.elm[sphere3_msh.elm.tag1 == 1004, :3])
        surf_1003 = np.unique(sphere3_msh.elm[sphere3_msh.elm.tag1 == 1003, :3])
        curv = sphere3_msh.gaussian_curvature()
        assert np.allclose(curv[surf_1005], (1/95**2), atol=1e-4)
        assert np.allclose(curv[surf_1004], (1/90**2), atol=1e-4)
        assert np.allclose(curv[surf_1003], (1/85**2), atol=1e-4)

        assert np.average(curv[surf_1005]) < np.average(curv[surf_1004])
        assert np.average(curv[surf_1004]) < np.average(curv[surf_1003])

    def test_node_volume(self, sphere3_msh):
        v = sphere3_msh.nodes_volumes_or_areas()
        tot_vol = np.sum(v.value)
        # high tolerance due to low resolution
        np.testing.assert_allclose(tot_vol, 4. / 3. * np.pi * 95**3, rtol=2e-1)

    def test_node_area(self, sphere3_msh):
        msh = sphere3_msh.crop_mesh(1005)
        v = msh.nodes_volumes_or_areas()
        tot_area = np.sum(v.value)
        # high tolerance due to low resolution
        assert v.field_name == 'areas'
        np.testing.assert_allclose(tot_area, 4. * np.pi * 95**2, rtol=2e-1)

    def tests_node_normal(self, sphere3_msh):
        normals = sphere3_msh.nodes_normals()
        outer_nodes = np.unique(
            sphere3_msh.elm.node_number_list[sphere3_msh.elm.tag1 == 1005, :3])
        # high tolerance due to low resolution
        np.testing.assert_allclose(normals[outer_nodes],
                                   sphere3_msh.nodes[outer_nodes] / 95,
                                   atol=1e-1, rtol=1e-1)

        normals = sphere3_msh.nodes_normals(smooth=3)
        # high tolerance due to low resolution
        np.testing.assert_allclose(normals[outer_nodes],
                                   sphere3_msh.nodes[outer_nodes] / 95,
                                   atol=1e-1, rtol=1e-1)

    def tests_node_area2(self, sphere3_msh):
        areas = sphere3_msh.nodes_areas()
        outer_nodes = np.unique(
            sphere3_msh.elm.node_number_list[sphere3_msh.elm.tag1 == 1005, :3])
        # high tolerance due to low resolution
        assert np.isclose(np.sum(areas[outer_nodes]), np.pi * 4. * 95 ** 2, rtol=1)

    '''
    def test_find_tetrahedron_with_points(self, sphere3_msh):
        X, Y, Z = np.meshgrid(np.linspace(-100, 100, 1000),
                              np.linspace(-40, 40, 100), [0])
        points = np.vstack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).T
        dist = np.linalg.norm(points, axis=1)

        points_outside = points[dist > 95]
        th_with_points = sphere3_msh.find_tetrahedron_with_points(points_outside)
        assert np.all(th_with_points == -1)

        points_inside = points[dist <= 94]
        th_with_points = sphere3_msh.find_tetrahedron_with_points(points_inside)
        assert np.all(th_with_points != -1)

        th_coords = \
            sphere3_msh.nodes[sphere3_msh.elm[th_with_points]]
        M = \
            np.transpose(th_coords[:, :3, :3] - th_coords[:, 3, None, :], (0, 2, 1))
        baricentric = np.linalg.solve(M, points_inside - th_coords[:, 3, :])
        eps = 1e-3
        assert np.all(baricentric >= 0 - eps)
        assert np.all(baricentric <= 1. + eps)
        assert np.all(np.sum(baricentric, axis=1) <= 1. + eps)
    '''

    def test_find_tetrahedron_with_points_non_convex(self, sphere3_msh):
        X, Y, Z = np.meshgrid(np.linspace(-100, 100, 100),
                              np.linspace(-40, 40, 100), [0])
        points = np.vstack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).T
        dist = np.linalg.norm(points, axis=1)

        msh = sphere3_msh.crop_mesh(5)
        points_outside = points[(dist > 95) + (dist < 89)]
        th_with_points, bar = msh.find_tetrahedron_with_points(points_outside)
        assert np.all(th_with_points == -1)
        assert np.allclose(bar, 0)

        points_inside = points[(dist <= 94) * (dist >= 91)]
        th_with_points, bar = msh.find_tetrahedron_with_points(points_inside)
        eps = 1e-3
        assert np.all(th_with_points != -1)
        assert np.all(bar >= 0 - eps)
        assert np.all(bar <= 1. + eps)
        th_coords = \
            msh.nodes[msh.elm[th_with_points]]
        assert np.allclose(np.einsum('ikj, ik -> ij', th_coords, bar), points_inside)

    def test_inside_volume(self, sphere3_msh):
        X, Y, Z = np.meshgrid(np.linspace(-100, 100, 100),
                              np.linspace(-40, 40, 10), [0])
        np.random.seed(0)
        points = np.vstack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).T
        points += np.random.random_sample(points.shape) - .5
        dist = np.linalg.norm(points, axis=1)

        msh = sphere3_msh.crop_mesh([4, 5])
        points_outside = points[(dist > 95) + (dist < 84)]
        inside = msh.test_inside_volume(points_outside)
        assert np.all(~inside)

        points_inside = points[(dist <= 94) * (dist >= 86)]
        inside = msh.test_inside_volume(points_inside)
        assert np.all(inside)

    def test_fix_node_ordering_unchanged(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        m.fix_th_node_ordering()
        assert np.all(m.elm.node_number_list == sphere3_msh.elm.node_number_list)

    def test_fix_node_ordering_changed(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        m.elm.node_number_list[:, 1] = sphere3_msh.elm.node_number_list[:, 2]
        m.elm.node_number_list[:, 2] = sphere3_msh.elm.node_number_list[:, 1]
        m.fix_th_node_ordering()
        th = m.nodes[m.elm.node_number_list[m.elm.elm_type == 4, :]]
        M = th[:, 1:] - th[:, 0, None]
        assert np.all(np.linalg.det(M) > 0)

    def test_fix_tr_node_ordering_unchanged(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        m.fix_tr_node_ordering()
        assert np.all(m.elm.node_number_list == sphere3_msh.elm.node_number_list)

    def test_fix_tr_node_ordering_changed(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        tmp = np.copy(m.elm.node_number_list[:, 1])
        m.elm.node_number_list[:, 1] = m.elm.node_number_list[:, 2]
        m.elm.node_number_list[:, 2] = tmp
        m.fix_tr_node_ordering()
        triangles = m.elm.elm_type == 2
        normal_sphere = m.elements_baricenters().value[triangles]
        normal_sphere /= np.linalg.norm(normal_sphere, axis=1)[:, None]
        dotp = np.sum(m.triangle_normals().value[triangles] * normal_sphere, axis=1)
        assert np.all(dotp > .9)


    def test_find_corresponding_tetrahedra(self, sphere3_msh):
        corrensponding = sphere3_msh.find_corresponding_tetrahedra()
        bar = sphere3_msh.elements_baricenters()
        direction = bar / np.linalg.norm(bar.value, axis=1)[:, None]
        tr_data = direction[sphere3_msh.elm.triangles]
        direction[sphere3_msh.elm.triangles] = 0
        assert np.allclose(direction[corrensponding], tr_data,
                           rtol=1e-1, atol=1e-1)

        corrensponding = sphere3_msh.find_corresponding_tetrahedra()
        # Try on a me
        m = sphere3_msh.crop_mesh([3, 4, 5, 1005])
        corrensponding = m.find_corresponding_tetrahedra()
        bar = m.elements_baricenters()
        direction = bar / np.linalg.norm(bar.value, axis=1)[:, None]
        tr_data = direction[m.elm.triangles]
        direction[m.elm.triangles] = 0
        assert np.allclose(direction[corrensponding], tr_data,
                           rtol=1e-1, atol=1e-1)

    def test_fix_surface_labels(self, sphere3_msh):
        msh = copy.deepcopy(sphere3_msh)
        msh.elm.tag1[msh.elm.elm_type == 2] -= 1000
        msh.elm.tag2[msh.elm.elm_type == 2] -= 1000
        msh.fix_surface_labels()
        assert np.all(msh.elm.tag1[msh.elm.elm_type == 2] > 1000)
        assert np.all(msh.elm.tag2[msh.elm.elm_type == 2] > 1000)

    def test_calc_matsimnibs(self, sphere3_msh):
        matsimnibs = sphere3_msh.calc_matsimnibs(
            [0., 0.,-95.], [0, 95., 0.], 1)
        assert np.allclose(matsimnibs[:3, :3], np.eye(3), atol=1e-2)
        assert np.allclose(matsimnibs[:3, 3], [0., 0., -96.], atol=1e-2)

    def test_intersect_segment(self, sphere3_msh):
        m = sphere3_msh.crop_mesh(1005)
        near = np.array([[120, 19, 32], [89, 142, 65]])
        far = -near
        idx, pos = m.intersect_segment(near, far)
        proj2surf = near/np.linalg.norm(near, axis=1)[:, None]*95
        assert np.all(np.linalg.norm(pos[[0, 2], :] + proj2surf, axis=1) < 0.3)
        assert np.all(np.linalg.norm(pos[[1, 3], :] - proj2surf, axis=1) < 0.3)
        assert np.all(np.linalg.norm(m.elements_baricenters()[idx[:, 1]] - pos, axis=1) < 3)

    def test_intersect_segment_no_hit(self, sphere3_msh):
        m = sphere3_msh.crop_mesh(1005)
        near = np.array([[120, 19, 32], [89, 142, 65]])
        far = .99 * near
        idx, pos = m.intersect_segment(near, far)
        assert len(idx) == 0
        assert len(pos) == 0

    def test_intersect_segment_getfarpoint(self):
        # generate cuboid
        vertices=np.array([[-1,-3,-5],[2,-3,-5],[2,4,-5],[-1,4,-5],
                           [-1,-3,6], [2,-3,6], [2,4,6], [-1,4,6]])
        hull = ConvexHull(vertices)
        elements=hull.simplices+1
        S=mesh_io.Msh(mesh_io.Nodes(vertices),mesh_io.Elements(triangles=elements))

        # generate random points and directions
        points=np.random.uniform(-10, 10, size=(100000,3))
        directions=np.random.uniform(-1, 1, size=(100000,3))

        idx,endpoints = S._intersect_segment_getfarpoint(points, directions)

        # test whether all intersection points are on the surface of the
        # internally used bounding box
        eps=0.01 # internal eps in _intersect_segment_getfarpoint to create ROI
        ROI=np.array([[-1-eps,2+eps],[-3-eps,4+eps],[-5-eps,6+eps]])

        eps=100*np.finfo(float).eps
        if len(idx)>0:
            nhits = (np.abs(endpoints[:,0] - ROI[0,0]) < eps).astype(int) + \
                    (np.abs(endpoints[:,0] - ROI[0,1]) < eps).astype(int) + \
                    (np.abs(endpoints[:,1] - ROI[1,0]) < eps).astype(int) + \
                    (np.abs(endpoints[:,1] - ROI[1,1]) < eps).astype(int) + \
                    (np.abs(endpoints[:,2] - ROI[2,0]) < eps).astype(int) + \
                    (np.abs(endpoints[:,2] - ROI[2,1]) < eps).astype(int)
            assert np.all(nhits == 1)

    def test_intersect_ray(self):
        # generate cuboid and fix triangle orientations
        vertices=np.array([[-1,-3,-5],[2,-3,-5],[2,4,-5],[-1,4,-5],
                           [-1,-3,6], [2,-3,6], [2,4,6], [-1,4,6]])
        hull = ConvexHull(vertices)
        elements=hull.simplices+1
        S=mesh_io.Msh(mesh_io.Nodes(vertices),
                      mesh_io.Elements(triangles=elements))

        normals = S.triangle_normals()[:]
        baricenters = S.elements_baricenters()[:]
        baricenters -= np.mean(baricenters,axis=0)
        dotp = np.einsum('ij, ij -> i', normals, baricenters)
        switch = dotp < 0
        buffer=S.elm.node_number_list[switch,1]
        S.elm.node_number_list[switch,1] = S.elm.node_number_list[switch,2]
        S.elm.node_number_list[switch,2] = buffer

        # generate random points and directions
        points=np.random.uniform(-10, 10, size=(100000,3))
        directions=np.random.uniform(-1, 1, size=(100000,3))

        idx,inters_pos = S.intersect_ray(points, directions)

        # test whether all intersection points are on the cuboid surface
        ROI=np.array([[-1,2],[-3,4],[-5,6]])
        eps=100*np.finfo(float).eps
        if len(idx)>0:
            nhits = (np.abs(inters_pos[:,0] - ROI[0,0]) < eps).astype(int) + \
                    (np.abs(inters_pos[:,0] - ROI[0,1]) < eps).astype(int) + \
                    (np.abs(inters_pos[:,1] - ROI[1,0]) < eps).astype(int) + \
                    (np.abs(inters_pos[:,1] - ROI[1,1]) < eps).astype(int) + \
                    (np.abs(inters_pos[:,2] - ROI[2,0]) < eps).astype(int) + \
                    (np.abs(inters_pos[:,2] - ROI[2,1]) < eps).astype(int)
            assert np.all(nhits == 1)

    def test_partition_skin_surface(self, sphere3_msh):

        mesh = copy.deepcopy(sphere3_msh)
        nn = mesh.nodes_normals()

        triangles = np.where(mesh.elm.tag1 == 1005)[0] + 1
        elm_idx = triangles[0]
        v = mesh.elm[elm_idx][0]

        v_tris = triangles[(mesh.elm.node_number_list[triangles-1, :3] == v).sum(1) > 0]
        v_to_move = np.unique(mesh.elm.node_number_list[v_tris-1, :3])

        f = np.full(v_to_move.shape, 30)

        # Update vertex positions
        mesh.nodes.node_coord[v_to_move-1] -= nn[v-1] * f[:, None]

        v_in_1, v_out_1, f_in_1, f_out_1 = mesh.partition_skin_surface()
        # v is considered outside even though it is surrounded by inside nodes
        v_in, v_out, f_in, f_out = mesh.partition_skin_surface(
            assume_single_outside_component=False
        )
        assert v in v_in_1
        assert np.isin(v_to_move, v_in_1).all(), "All updated nodes should be considered 'inside'."
        assert v not in v_in
        # Currently, f_in/f_out is the same irrespective of the call
        assert np.all(f_in == f_in_1)
        assert np.all(f_out == f_out_1)
        assert np.all(np.isin(mesh.elm[f_in_1][:, :3], v_to_move).sum(1) >= 2)

    def test_elm2node_matrix(self, sphere3_msh):
        m = sphere3_msh.crop_mesh(elm_type=4)
        M = m.elm2node_matrix()
        x = m.elements_baricenters().value[:, 0]
        assert np.allclose(
            M.dot(x), m.nodes.node_coord[:, 0], atol=3, rtol=5e-1)

    def test_elm2node_matrix_crop(self, sphere3_msh):
        m = sphere3_msh.crop_mesh(elm_type=4)
        M = m.elm2node_matrix(m.elm.elm_number[m.elm.tag1 == 3])
        x = m.elements_baricenters().value[:, 0]
        nodes_r = np.linalg.norm(m.nodes.node_coord, axis=1)
        assert np.allclose(
            M.dot(x)[nodes_r < 85],
            m.nodes.node_coord[nodes_r < 85, 0],
            atol=3, rtol=5e-1)
        assert np.allclose(
            M.dot(x)[nodes_r > 85.1], 0)

    def test_elm2node_matrix_surf(self, sphere3_msh):
        m = sphere3_msh.crop_mesh(elm_type=2)
        M = m.elm2node_matrix()
        x = m.elements_baricenters().value[:, 0]
        assert np.allclose(M.dot(x), m.nodes.node_coord[:, 0], atol=3)

    def test_elm2node_matrix_mixed(self, sphere3_msh):
        m = sphere3_msh.crop_mesh([3, 1003, 1005])
        M = m.elm2node_matrix()
        x_node = m.nodes.node_coord[:, 0]
        x_elms = m.elements_baricenters().value[:, 0]
        R_node = np.linalg.norm(m.nodes.node_coord, axis=1)
        R_elm = np.linalg.norm(m.elements_baricenters()[:], axis=1)
        region = R_node < 84
        assert np.allclose(M.dot(x_elms)[region], x_node[region], atol=1e-3)
        region = (R_node > 84) * (R_node < 86)
        assert np.allclose(M.dot(x_elms)[region], x_node[region], atol=10)
        region = R_node > 86
        assert np.allclose(M.dot(x_elms)[region], x_node[region], atol=3)


    def test_interp_matrix(self, sphere3_msh):
        m = sphere3_msh.crop_mesh(elm_type=4)
        m_out = sphere3_msh.crop_mesh(1005)
        interp_points = m_out.elements_baricenters().value / 95. * 51.2
        M = m.interp_matrix(interp_points)
        x = m.nodes.node_coord[:, 0]
        assert np.allclose(
            M.dot(x), interp_points[:, 0], rtol=1e-3)

    def test_interp_matrix_out(self, sphere3_msh):
        m = sphere3_msh.crop_mesh(elm_type=4)
        m_out = sphere3_msh.crop_mesh(1005)
        interp_points = m_out.elements_baricenters().value * 1.01
        M = m.interp_matrix(interp_points)
        x = m.nodes.node_coord[:, 0]
        assert np.all(np.isnan(M.dot(x)))

    def test_interp_matrix_out_nn(self, sphere3_msh):
        m = sphere3_msh.crop_mesh(elm_type=4)
        m_out = sphere3_msh.crop_mesh(1005)
        interp_points = m_out.nodes.node_coord * 1.1
        M = m.interp_matrix(interp_points, 'nearest')
        x = m.nodes.node_coord[:, 0]
        assert np.allclose(M.dot(x), interp_points[:, 0], atol=1, rtol=1e-1)

    def test_interp_matrix_elements(self, sphere3_msh):
        m = sphere3_msh.crop_mesh(elm_type=4)
        m_out = sphere3_msh.crop_mesh(1005)
        interp_points = m_out.elements_baricenters().value / 95. * 51.2
        M = m.interp_matrix(interp_points, element_wise=True)
        x = m.elements_baricenters().value[:, 0]
        assert np.allclose(M.dot(x), interp_points[:, 0], rtol=1e-3)

    @pytest.mark.parametrize("element_wise", [False, True])
    def test_interp_matrix_th(self, element_wise, sphere3_msh):
        m = sphere3_msh.crop_mesh(elm_type=4)
        m_out = sphere3_msh.crop_mesh(1005)
        interp_points = np.vstack([
            m_out.elements_baricenters().value / 95. * 50.,
            m_out.elements_baricenters().value / 95. * 87.5])
        inside = np.ones(len(interp_points), dtype=bool)
        inside[m_out.elm.nr:] = False
        M = m.interp_matrix(interp_points,
                            th_indices=m.elm.elm_number[np.isin(m.elm.tag1, [3, 4, 5])],
                            element_wise=element_wise)
        if element_wise:
            x = m.elements_baricenters().value[:, 0]
        else:
            x = m.nodes.node_coord[:, 0]
        assert np.allclose(M.dot(x), interp_points[:, 0], atol=1, rtol=1e-1)

    def test_find_shared_nodes(self, sphere3_msh):
        shared_nodes = sphere3_msh.find_shared_nodes([3, 4])
        surf_nodes = np.unique(sphere3_msh.elm[sphere3_msh.elm.tag1==1003, :3])
        assert np.all(shared_nodes == surf_nodes)

    def test_find_shared_nodes_tr(self, sphere3_msh):
        shared_nodes = sphere3_msh.find_shared_nodes([3, 1003])
        surf_nodes = np.unique(sphere3_msh.elm[sphere3_msh.elm.tag1==1003, :3])
        print(shared_nodes)
        assert np.all(shared_nodes == surf_nodes)

    def test_find_shared_nodes_empty(self, sphere3_msh):
        shared_nodes = sphere3_msh.find_shared_nodes([3, 5])
        assert len(shared_nodes) == 0

    def test_add_node_field_value(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        m.add_node_field(m.nodes.node_coord, 'node_coord')
        assert np.allclose(m.nodedata[0].value, m.nodes.node_coord)
        assert m.nodedata[0].field_name == 'node_coord'
        assert m.nodedata[0].mesh == m
        assert isinstance(m.nodedata[0], mesh_io.NodeData)

    def test_add_node_field_nd(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        nd = mesh_io.NodeData(m.nodes.node_coord, 'a')
        m.add_node_field(nd, 'node_coord')
        assert np.allclose(m.nodedata[0].value, m.nodes.node_coord)
        assert m.nodedata[0].field_name == 'node_coord'
        assert m.nodedata[0].mesh == m
        assert isinstance(m.nodedata[0], mesh_io.NodeData)

    def test_add_node_field_error(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        with pytest.raises(ValueError):
            m.add_node_field(m.nodes.node_coord[1:], 'node_coord')

    def test_add_elm_field_value(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        m.add_element_field(m.elm.tag1, 'tags')
        assert np.allclose(m.elmdata[0].value, m.elm.tag1)
        assert m.elmdata[0].field_name == 'tags'
        assert m.elmdata[0].mesh == m
        assert isinstance(m.elmdata[0], mesh_io.ElementData)

    def test_add_element_field_nd(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        ed = mesh_io.ElementData(m.elm.tag1, 'a')
        m.add_element_field(ed, 'tags')
        assert m.elmdata[0].field_name == 'tags'
        assert m.elmdata[0].mesh == m
        assert isinstance(m.elmdata[0], mesh_io.ElementData)

    def test_add_element_field_error(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        with pytest.raises(ValueError):
            m.add_element_field(m.elm.tag1[1:], 'tags')

    def test_reconstruct_surfaces(self, sphere3_msh):
        m = sphere3_msh.crop_mesh(elm_type=4)
        m.reconstruct_surfaces()
        c = m.elements_baricenters()
        assert np.allclose(np.linalg.norm(c[m.elm.tag1 == 1003], axis=1), 85, rtol=1e-2)
        assert np.all(
            np.isclose(np.linalg.norm(c[m.elm.tag1 == 1004], axis=1), 85, rtol=1e-2) +
            np.isclose(np.linalg.norm(c[m.elm.tag1 == 1004], axis=1), 90, rtol=1e-2)
            )
        assert np.all(
            np.isclose(np.linalg.norm(c[m.elm.tag1 == 1005], axis=1), 90, rtol=1e-2) +
            np.isclose(np.linalg.norm(c[m.elm.tag1 == 1005], axis=1), 95, rtol=1e-2)
            )

    def test_reconstruct_surfaces_tag(self, sphere3_msh):
        m = sphere3_msh.crop_mesh(elm_type=4)
        m.reconstruct_surfaces(tags=[4])
        assert ~np.any(np.isin(m.elm.tag1, [1003, 1005]))

    def test_reconstruct_unique_surface(self, sphere3_msh):
        m2=sphere3_msh.crop_mesh(elm_type=2)
        m=sphere3_msh.crop_mesh(elm_type=4)
        m.reconstruct_unique_surface()
        m = m.crop_mesh(elm_type=2)
        m_tags, m_cts = np.unique(m.elm.tag1,return_counts = True)
        m2_tags, m2_cts = np.unique(m2.elm.tag1,return_counts = True)
        assert m.elm.nr == m2.elm.nr
        assert np.all(m2_tags == m_tags)
        assert np.all(m2_cts == m_cts)

    def test_smooth_calc_gamma(self):
        tetrahedra=np.array([0, 1, 2, 3],dtype=np.uint)
        nodes = np.array(
            [[1./3. * np.sqrt(3), 0, 0],
             [-1./6. * np.sqrt(3), 1./2., 0],
             [-1./6. * np.sqrt(3), -1./2., 0],
             [0, 0, 1./3. * np.sqrt(6)],
             [0, 0, 1./3. * np.sqrt(6)]], dtype=float)
        gamma = mesh_io.cython_msh.calc_gamma(nodes, tetrahedra)
        assert np.isclose(gamma, 1., rtol=1e-3)

    def test_gamma_metric(self):
        msh = mesh_io.Msh()
        # 2 regular tetrahedron of edge size 1
        msh.elm = mesh_io.Elements(
            tetrahedra=np.array(
                [[1, 2, 3, 4],
                 [1, 2, 5, 4]], dtype=int))
        msh.elm.elm_type = np.array([4, 4])
        msh.nodes = mesh_io.Nodes(np.array(
            [[1./3. * np.sqrt(3), 0, 0],
             [-1./6. * np.sqrt(3), 1./2., 0],
             [0, 0, 1./3. * np.sqrt(6)],
             [-1./6. * np.sqrt(3), -1./2., 0],
             [0, 0, 1./3. * np.sqrt(6)]], dtype=float))
        gamma = msh.gamma_metric()
        assert np.allclose(gamma[:], 1, rtol=1e-3)


    def test_smooth_without_th(self, sphere3_msh):
        mesh = sphere3_msh.crop_mesh(elm_type=2)
        tr_nodes = np.unique(mesh.elm[mesh.elm.tag1 == 1003, :3])
        mesh.nodes.node_coord[tr_nodes - 1] += \
            np.random.standard_normal((len(tr_nodes), 3)) * 1
        curvature_before = mesh.gaussian_curvature()[tr_nodes]
        mesh.smooth_surfaces(10)
        curvature_after = mesh.gaussian_curvature()[tr_nodes]
        assert np.std(curvature_after) < np.std(curvature_before)

    def test_smooth_with_tetrahedra(self, sphere3_msh):
        mesh = copy.deepcopy(sphere3_msh)
        tr_nodes = np.unique(mesh.elm[mesh.elm.tag1 == 1003, :3])
        mesh.nodes.node_coord[tr_nodes - 1] += \
            np.random.standard_normal((len(tr_nodes), 3)) * 1
        curvature_before = mesh.gaussian_curvature()[tr_nodes]
        mesh.smooth_surfaces(10, max_gamma=100)
        curvature_after = mesh.gaussian_curvature()[tr_nodes]
        assert np.std(curvature_after) < np.std(curvature_before)

    def test_smooth_with_tag(self, sphere3_msh):
        mesh = sphere3_msh.crop_mesh(elm_type=2)
        tr_nodes = np.unique(mesh.elm[mesh.elm.tag1 == 1003, :3])
        mesh.nodes.node_coord[tr_nodes - 1] += \
            np.random.standard_normal((len(tr_nodes), 3)) * 1
        curvature_before = mesh.gaussian_curvature()
        mask_nodes = np.zeros(mesh.nodes.nr, dtype=bool)
        mask_nodes[tr_nodes - 1] = True
        mesh.smooth_surfaces(10, tags=1003, max_gamma=100)
        curvature_after = mesh.gaussian_curvature()
        assert np.std(curvature_after[mask_nodes]) < np.std(curvature_before[mask_nodes])
        assert np.allclose(curvature_after[~mask_nodes], curvature_before[~mask_nodes])

    def test_smooth_quality(self, sphere3_msh):
        mesh = copy.deepcopy(sphere3_msh)
        tr_nodes = np.unique(mesh.elm[mesh.elm.tag1 == 1003, :3])
        mesh.nodes.node_coord[tr_nodes - 1] += \
            np.random.standard_normal((len(tr_nodes), 3)) * 0.1
        mesh_before = copy.deepcopy(mesh)
        mesh.smooth_surfaces(10, max_gamma=3)
        curvature_after = mesh.gaussian_curvature()
        low_q = mesh_before.gamma_metric()[:] >= 3
        assert np.all(mesh_before.gamma_metric()[~low_q] < 3)

    def test_split_tets_along_line(self, sphere3_msh):
        m=sphere3_msh.crop_mesh(elm_type = 4)
        n_nodes_pre = m.nodes.nr
        n_tets_pre = m.elm.nr
        idx_tet1, idx_tet2 = m.split_tets_along_line(1246,2337,return_tetindices=True)
        assert m.nodes.nr == n_nodes_pre+1
        assert m.elm.nr == n_tets_pre + len(idx_tet1)


class TestData:
    def test_read_hdf5_data_matrix_row(self):
        E = np.arange(600).reshape(10, 20, 3)
        with h5py.File('tmp.hdf5', 'w') as f:
            f.create_dataset('leadfield/E', data=E)
        data = mesh_io.Data.read_hdf5_data_matrix_row('tmp.hdf5', 'leadfield/E', 1)
        np.testing.assert_equal(data.value, E[1])

        os.remove('tmp.hdf5')

    def test_inepolate_to_surface(self, sphere3_msh):
        field = mesh_io.NodeData(sphere3_msh.nodes.node_coord[:, 0], mesh=sphere3_msh)
        surface = sphere3_msh.crop_mesh(1003)
        interp = field.interpolate_to_surface(surface)
        assert np.allclose(interp.value, surface.nodes.node_coord[:, 0])

    @pytest.mark.parametrize('n_comp', [1, 3])
    @pytest.mark.parametrize('type_', ['node', 'element'])
    def test_mean_field(self, type_, n_comp, sphere3_msh):
        m = sphere3_msh.crop_mesh(elm_type=4)
        if type_ == 'node':
            field = mesh_io.NodeData(np.ones((m.nodes.nr, n_comp)),
                                  mesh=m)
        if type_ == 'element':
            field = mesh_io.ElementData(np.ones((m.elm.nr, n_comp)),
                                     mesh=m)
        eff_area = field.mean_field_norm()
        assert np.isclose(eff_area, np.sqrt(n_comp))


    @pytest.mark.parametrize('n_comp', [1, 3])
    def test_get_percentile(self, n_comp, sphere3_msh):
        m = sphere3_msh.crop_mesh(elm_type=4)
        field = mesh_io.NodeData(
            np.random.rand(m.nodes.nr, n_comp), mesh=m)
        prc = field.get_percentiles(90)
        vols = m.nodes_volumes_or_areas()[:]
        if n_comp == 3:
            v = np.linalg.norm(field[:], axis=1)
        else:
            v = np.squeeze(field[:])
        assert np.isclose(np.sum(vols[v > prc]),
                          np.sum(vols) / 10, rtol=1e-1)

    def test_get_percentile_elm(self, sphere3_msh):
        m = sphere3_msh.crop_mesh(elm_type=4)
        field = mesh_io.ElementData(
            np.random.rand(m.elm.nr, 3),
            mesh=m)
        prc = field.get_percentiles([90, 95])
        vols = m.elements_volumes_and_areas()[:]
        norm = np.linalg.norm(field[:], axis=1)
        assert np.isclose(np.sum(vols[norm > prc[0]]),
                          np.sum(vols) / 10, rtol=1e-1)
        assert np.isclose(np.sum(vols[norm > prc[1]]),
                          np.sum(vols) / 20, rtol=1e-1)

    def test_get_percentile_roi(self, sphere3_msh):
        m = sphere3_msh.crop_mesh(elm_type=4)
        roi = m.elm.tag1 == 3
        field = mesh_io.ElementData(
            np.random.rand(m.elm.nr, 3),
            mesh=m)
        prc = field.get_percentiles([90, 95], roi=roi)
        vols = m.elements_volumes_and_areas()[roi]
        norm = np.linalg.norm(field[roi], axis=1)
        assert np.isclose(np.sum(vols[norm > prc[0]]),
                          np.sum(vols) / 10, rtol=1e-1)
        assert np.isclose(np.sum(vols[norm > prc[1]]),
                          np.sum(vols) / 20, rtol=1e-1)


    def test_focality(self, sphere3_msh):
        m = sphere3_msh.crop_mesh(elm_type=4)
        field = mesh_io.ElementData(
            np.random.rand(m.elm.nr),
            mesh=m)
        foc = field.get_focality(cuttofs=[80, 90], peak_percentile=99.9)
        vols = m.elements_volumes_and_areas()[:]
        tot_vol = np.sum(vols)
        assert np.isclose(foc[0], tot_vol * .2, rtol=1e-1)
        assert np.isclose(foc[1], tot_vol * .1, rtol=1e-1)

    def test_summary(self, sphere3_msh):
        m = sphere3_msh.crop_mesh(elm_type=4)
        field = mesh_io.ElementData(
            np.random.rand(m.elm.nr),
            mesh=m)
        field.summary(units='A.u')



class TestElmData:
    def test_write_elmdata_scalar(self, sphere3_msh):
        elm_data = mesh_io.ElementData(
            sphere3_msh.elm.tag1.astype('float64'), 'elm_data_scalar')
        tmp = copy.deepcopy(sphere3_msh)
        tmp.elmdata.append(elm_data)
        mesh_io.write_msh(tmp, 'tmp.msh')
        tmp = mesh_io.read_msh('tmp.msh')
        os.remove('tmp.msh')
        np.testing.assert_array_equal(
            tmp.elmdata[0].value, sphere3_msh.elm.tag1)

    def test_write_elmdata_vect(self, sphere3_msh):
        v = np.vstack(
            (sphere3_msh.elm.tag1,
             sphere3_msh.elm.tag1,
             sphere3_msh.elm.tag1)).T
        elm_data = mesh_io.ElementData(v, 'elm_data_vectorial')
        tmp = copy.deepcopy(sphere3_msh)
        tmp.elmdata.append(elm_data)
        mesh_io.write_msh(tmp, 'tmp.msh')
        tmp = mesh_io.read_msh('tmp.msh')
        os.remove('tmp.msh')
        np.testing.assert_array_equal(tmp.elmdata[0].value.astype('int32'), v)

    def test_elm_data2node_data_scalar(self, sphere3_msh):
        x = sphere3_msh.elements_baricenters().value[:, 0]
        elm_data = mesh_io.ElementData(x, mesh=sphere3_msh)
        node_data = elm_data.elm_data2node_data()
        assert np.allclose(node_data.value,
                           sphere3_msh.nodes.node_coord[:, 0],
                           atol=3, rtol=5e-1)

    def test_elm_data2node_data_vectorial(self, sphere3_msh):
        x = sphere3_msh.elements_baricenters().value
        elm_data = mesh_io.ElementData(x, mesh=sphere3_msh)
        node_data = elm_data.elm_data2node_data()
        assert np.allclose(node_data.value,
                           sphere3_msh.nodes.node_coord,
                           atol=4, rtol=5e-1)


    def test_interpolate_scattered(self, sphere3_msh):
        bar = sphere3_msh.elements_baricenters()
        bar.mesh = sphere3_msh
        bar.value = bar.value[:, 0]
        interp_points = sphere3_msh.crop_mesh(elm_type=4).elements_baricenters().value
        interp = bar.interpolate_scattered(interp_points)
        assert np.allclose(interp, interp_points[:, 0], atol=2, rtol=1e-1)

    def test_interpolate_scattered_vect(self, sphere3_msh):
        bar = sphere3_msh.elements_baricenters()
        bar.mesh = sphere3_msh
        interp_points = \
            sphere3_msh.crop_mesh(elm_type=4).elements_baricenters().value[:10]
        interp = bar.interpolate_scattered(interp_points)
        assert np.allclose(interp, interp_points, atol=5, rtol=1e-1)

    def test_interpolate_scattered_continuous(self, sphere3_msh):
        bar = sphere3_msh.elements_baricenters()
        bar.mesh = sphere3_msh
        bar.value = bar.value[:, 0]
        interp_points = sphere3_msh.crop_mesh(elm_type=4).elements_baricenters().value
        interp = bar.interpolate_scattered(interp_points, continuous=True)
        assert np.allclose(interp, interp_points[:, 0], atol=2, rtol=1e-2)


    def test_interpolate_scattered_assign(self, sphere3_msh):
        bar = sphere3_msh.elements_baricenters()
        bar.mesh = sphere3_msh
        interp_points = sphere3_msh.nodes.node_coord[:10]
        interp = bar.interpolate_scattered(interp_points, method='assign')
        assert np.allclose(interp, interp_points, atol=5, rtol=1e-1)

    def test_interpolate_scattered_th(self, sphere3_msh):
        bar = sphere3_msh.elements_baricenters()
        bar.mesh = sphere3_msh
        interp_points = sphere3_msh.nodes.node_coord[:10]
        th_indices = sphere3_msh.elm.elm_number[np.isin(
            sphere3_msh.elm.tag1, [3, 4, 5])]
        interp = bar.interpolate_scattered(
            interp_points, th_indices=th_indices)
        assert np.allclose(interp, interp_points, atol=5, rtol=1e-1)

    def test_interpolate_grid_const_nn(self, sphere3_msh):
        data = sphere3_msh.elm.tag1
        f = mesh_io.ElementData(data, mesh=sphere3_msh)
        n = (200, 10, 1)
        affine = np.array([[1, 0, 0, -100.5],
                           [0, 1, 0, -5],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        interp = f.interpolate_to_grid(n, affine, method='assign')
        '''
        import matplotlib.pyplot as plt
        plt.imshow(np.squeeze(interp))
        plt.colorbar()
        plt.show()
        assert False
        '''
        assert np.isclose(interp[100, 5, 0], 3)
        assert np.isclose(interp[187, 5, 0], 4)
        assert np.isclose(interp[193, 5, 0], 5)
        assert np.isclose(interp[198, 5, 0], 0)

    def test_interpolate_grid_vec_nn(self, sphere3_msh):
        data = np.vstack([sphere3_msh.elm.tag1,
                          sphere3_msh.elm.tag1 + 10,
                          sphere3_msh.elm.tag1 + 20]).T
        f = mesh_io.ElementData(data, mesh=sphere3_msh)
        n = (200, 10, 1)
        affine = np.array([[1, 0, 0, -100],
                           [0, 1, 0, -5],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        interp = f.interpolate_to_grid(n, affine, method='assign')
        assert np.allclose(interp[100, 5, 0, :], [3, 13, 23])
        assert np.allclose(interp[187, 5, 0, :], [4, 14, 24])
        assert np.allclose(interp[193, 5, 0, :], [5, 15, 25])
        assert np.allclose(interp[198, 5, 0, :], [0, 0, 0])

    def test_interpolate_grid_size_nn(self, sphere3_msh):
        data = sphere3_msh.elm.tag1
        f = mesh_io.ElementData(data, mesh=sphere3_msh)
        n = (100, 5, 1)
        affine = np.array([[2, 0, 0, -100],
                           [0, 2, 0, -5],
                           [0, 0, 2, 0],
                           [0, 0, 0, 1]], dtype=float)
        interp = f.interpolate_to_grid(n, affine, method='assign')
        assert np.isclose(interp[50, 2, 0], 3)
        assert np.isclose(interp[93, 2, 0], 4)
        assert np.isclose(interp[96, 2, 0], 5)
        assert np.isclose(interp[98, 2, 0], 0)

    def test_interpolate_grid_rotate_nn(self, sphere3_msh):
        data = np.zeros(sphere3_msh.elm.nr)
        b = sphere3_msh.elements_baricenters().value
        f = mesh_io.ElementData(data, mesh=sphere3_msh)
        # Assign quadrant numbers
        f.value[(b[:, 0] > 0) * (b[:, 1] > 0)] = 1.
        f.value[(b[:, 0] < 0) * (b[:, 1] > 0)] = 2.
        f.value[(b[:, 0] < 0) * (b[:, 1] < 0)] = 3.
        f.value[(b[:, 0] > 0) * (b[:, 1] < 0)] = 4.
        n = (200, 200, 1)
        affine = np.array([[np.cos(np.pi/4.), np.sin(np.pi/4.), 0, -141],
                           [-np.sin(np.pi/4.), np.cos(np.pi/4.), 0, 0],
                           [0, 0, 1, .5],
                           [0, 0, 0, 1]], dtype=float)
        interp = f.interpolate_to_grid(n, affine, method='assign')
        '''
        import matplotlib.pyplot as plt
        plt.imshow(np.squeeze(interp))
        plt.colorbar()
        plt.show()
        '''
        assert np.isclose(interp[190, 100, 0], 4)
        assert np.isclose(interp[100, 190, 0], 1)
        assert np.isclose(interp[10, 100, 0], 2)
        assert np.isclose(interp[100, 10, 0], 3)

    def test_interpolate_grid_rotate_nodedata(self, sphere3_msh):
        data = np.zeros(sphere3_msh.nodes.nr)
        b = sphere3_msh.nodes.node_coord.copy()
        f = mesh_io.NodeData(data, mesh=sphere3_msh)
        # Assign quadrant numbers
        f.value[(b[:, 0] >= 0) * (b[:, 1] >= 0)] = 1.
        f.value[(b[:, 0] <= 0) * (b[:, 1] >= 0)] = 2.
        f.value[(b[:, 0] <= 0) * (b[:, 1] <= 0)] = 3.
        f.value[(b[:, 0] >= 0) * (b[:, 1] <= 0)] = 4.
        n = (200, 200, 1)
        affine = np.array([[np.cos(np.pi/4.), np.sin(np.pi/4.), 0, -141],
                           [-np.sin(np.pi/4.), np.cos(np.pi/4.), 0, 0],
                           [0, 0, 1, .5],
                           [0, 0, 0, 1]], dtype=float)
        interp = f.interpolate_to_grid(n, affine)
        '''
        import matplotlib.pyplot as plt
        plt.imshow(np.squeeze(interp), interpolation='nearest')
        plt.colorbar()
        plt.show()
        '''
        assert np.isclose(interp[190, 100, 0], 4)
        assert np.isclose(interp[100, 190, 0], 1)
        assert np.isclose(interp[10, 100, 0], 2)
        assert np.isclose(interp[100, 10, 0], 3)

    def test_interpolate_grid_elmdata_linear(self, sphere3_msh):
        data = sphere3_msh.elements_baricenters().value[:, 0]
        f = mesh_io.ElementData(data, mesh=sphere3_msh)
        n = (130, 130, 1)
        affine = np.array([[1, 0, 0, -65],
                           [0, 1, 0, -65],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        X, _ = np.meshgrid(np.arange(130), np.arange(130), indexing='ij')
        interp = f.interpolate_to_grid(n, affine, method='linear', continuous=True)
        '''
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(np.squeeze(interp))
        plt.colorbar()
        plt.show()
        '''
        assert np.allclose(interp[:, :, 0], X - 64.5, atol=1)

    def test_interpolate_grid_elmdata_dicontinuous(self, sphere3_msh):
        data = sphere3_msh.elm.tag1
        f = mesh_io.ElementData(data, mesh=sphere3_msh)
        n = (200, 130, 1)
        affine = np.array([[1, 0, 0, -100.1],
                           [0,-1, 0, 65.1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        interp = f.interpolate_to_grid(n, affine, method='linear', continuous=False)
        '''
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(np.squeeze(interp))
        plt.colorbar()
        plt.show()
        '''
        assert np.allclose(interp[6:10, 65, 0], 5, atol=1e-1)
        assert np.allclose(interp[11:15, 65, 0], 4, atol=1e-1)
        assert np.allclose(interp[16:100, 65, 0], 3, atol=1e-1)

    def test_to_affine_grid(self, sphere3_msh):
        import nibabel
        data = sphere3_msh.elm.tag1
        f = mesh_io.ElementData(data, mesh=sphere3_msh)
        affine = np.array([[1, 0, 0, -100.5],
                           [0, 1, 0, -5],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        img = nibabel.Nifti1Pair(np.zeros((200, 10, 1)), affine)
        tempf = tempfile.NamedTemporaryFile(suffix='.nii', delete=False)
        fn = tempf.name
        tempf.close()
        nibabel.save(img, tempf.name)
        affine = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        tempf2 = tempfile.NamedTemporaryFile(suffix='.mat', delete=False)
        fn2 = tempf2.name
        tempf2.close()
        np.savetxt(tempf2.name, affine)
        interp = f.to_deformed_grid(fn2, fn, method='assign')
        interp = np.asanyarray(interp.dataobj)
        assert np.isclose(interp[100, 5, 0], 3)
        assert np.isclose(interp[187, 5, 0], 4)
        assert np.isclose(interp[193, 5, 0], 5)
        assert np.isclose(interp[198, 5, 0], 0)


    def test_to_nonlinear_grid(self, sphere3_msh):
        import nibabel
        data = sphere3_msh.elm.tag1
        f = mesh_io.ElementData(data, mesh=sphere3_msh)
        affine = np.array([[1, 0, 0, -100.5],
                           [0, 1, 0, -5],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        x, y, z = np.meshgrid(np.arange(-100, 100),
                              np.arange(-5, 5),
                              np.arange(0, 1),
                              indexing='ij')
        nonl_transform = np.concatenate(
            (x[..., None], y[..., None], z[..., None]), axis=3).astype(float)
        img = nibabel.Nifti1Pair(nonl_transform, affine)
        tempf = tempfile.NamedTemporaryFile(suffix='.nii', delete=False)
        fn = tempf.name
        tempf.close()
        nibabel.save(img, fn)
        interp = f.to_deformed_grid(fn, fn, method='assign', fix_boundary_zeros=False)
        interp = np.asanyarray(interp.dataobj)
        assert np.isclose(interp[100, 5, 0], 3)
        assert np.isclose(interp[187, 5, 0], 4)
        assert np.isclose(interp[193, 5, 0], 5)
        assert np.isclose(interp[198, 5, 0], 0)

    def test_to_nonlinear_grid_crop(self, sphere3_msh):
        import nibabel
        data = sphere3_msh.elm.tag1
        f = mesh_io.ElementData(data, mesh=sphere3_msh)
        affine = np.array([[1, 0, 0, -100.5],
                           [0, 1, 0, -5],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        x, y, z = np.meshgrid(np.arange(-100, 100),
                              np.arange(-5, 5),
                              np.arange(0, 1),
                              indexing='ij')
        nonl_transform = np.concatenate(
            (x[..., None], y[..., None], z[..., None]), axis=3).astype(float)
        img = nibabel.Nifti1Pair(nonl_transform, affine)
        tempf = tempfile.NamedTemporaryFile(suffix='.nii', delete=False)
        fn = tempf.name
        tempf.close()
        nibabel.save(img, fn)
        interp = f.to_deformed_grid(fn, fn, tags=3, method='assign', fix_boundary_zeros=False)
        interp = np.asanyarray(interp.dataobj)
        assert np.isclose(interp[100, 5, 0], 3)
        assert np.isclose(interp[187, 5, 0], 0)
        assert np.isclose(interp[193, 5, 0], 0)
        assert np.isclose(interp[198, 5, 0], 0)

    def test_to_nonlinear_grid_linear_interp(self, sphere3_msh):
        import nibabel
        data = sphere3_msh.elements_baricenters().value[:, 0]
        f = mesh_io.ElementData(data, mesh=sphere3_msh)
        affine = np.array([[1, 0, 0, -100.5],
                           [0, 1, 0, -5],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        x, y, z = np.meshgrid(np.arange(-100, 100),
                              np.arange(-5, 5),
                              np.arange(0, 1),
                              indexing='ij')
        nonl_transform = np.concatenate(
            (x[..., None], y[..., None], z[..., None]), axis=3).astype(float)
        img = nibabel.Nifti1Pair(nonl_transform, affine)
        tempf = tempfile.NamedTemporaryFile(suffix='.nii',delete=False)
        fn = tempf.name
        tempf.close()
        nibabel.save(img, fn)
        interp = f.to_deformed_grid(fn, fn, order=1,
                                    method='linear', continuous=True)
        interp = np.asanyarray(interp.dataobj)
        '''
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(np.squeeze(interp))
        plt.colorbar()
        plt.show()
        '''
        assert np.isclose(interp[150, 5, 0], 50, atol=1e-2)
        assert np.isclose(interp[190, 5, 0], 90, atol=1e-1)
        assert np.isclose(interp[191, 5, 0], 91, atol=5e-1)
        assert np.isclose(interp[198, 5, 0], 0)


    def test_to_nonlinear_grid_nodedata(self, sphere3_msh):
        import nibabel
        data = sphere3_msh.nodes.node_coord[:, 0]
        f = mesh_io.NodeData(data, mesh=sphere3_msh)
        affine = np.array([[1, 0, 0, -100.5],
                           [0, 1, 0, -5],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        x, y, z = np.meshgrid(np.arange(-100, 100),
                              np.arange(-5, 5),
                              np.arange(0, 1),
                              indexing='ij')
        nonl_transform = np.concatenate(
            (x[..., None], y[..., None], z[..., None]), axis=3).astype(float)
        img = nibabel.Nifti1Pair(nonl_transform, affine)
        tempf = tempfile.NamedTemporaryFile(suffix='.nii', delete=False)
        fn = tempf.name
        tempf.close()
        nibabel.save(img, fn)
        interp = f.to_deformed_grid(fn, fn, tags=3, order=1)
        interp = np.asanyarray(interp.dataobj)
        '''
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(np.squeeze(interp))
        plt.colorbar()
        plt.show()
        '''
        assert np.isclose(interp[150, 5, 0], 50)
        assert np.isclose(interp[190, 5, 0], 0)
        assert np.isclose(interp[193, 5, 0], 0)
        assert np.isclose(interp[198, 5, 0], 0)


    def test_assign_triangle_values(self, sphere3_msh):
        data = sphere3_msh.elements_baricenters()
        data = data / np.linalg.norm(data.value, axis=1)[:, None]
        tr_data = data[sphere3_msh.elm.triangles]
        data[sphere3_msh.elm.triangles] = 0
        data.assign_triangle_values()
        assert np.allclose(data[sphere3_msh.elm.triangles], tr_data,
                           rtol=1e-1, atol=1e-1)

    def test_calc_flux(self, sphere3_msh):
        elmdata = mesh_io.ElementData(np.tile([1., 0., 0.], (sphere3_msh.elm.nr, 1)),
                                   mesh=sphere3_msh)
        triangles = sphere3_msh.elm.elm_number[sphere3_msh.elm.tag1 == 1003]
        flux = elmdata.calc_flux(triangles)
        assert np.isclose(flux, 0, atol=1e-1)

    def test_calc_flux_radial(self, sphere3_msh):
        elmdata = sphere3_msh.elements_baricenters()
        triangles = sphere3_msh.elm.elm_number[sphere3_msh.elm.tag1 == 1003]
        flux = elmdata.calc_flux(triangles)
        # Divergence theorem
        assert np.isclose(flux, 85 ** 3 * 4 * np.pi, rtol=1e-2)

    def test_from_data_grid(self, sphere3_msh):
        X, _, _ = np.meshgrid(np.arange(-100, 100),
                              np.arange(-100, 100),
                              np.arange(-100, 100),
                              indexing='ij')
        X = X.astype(float)
        affine = np.array([[1, 0, 0, -100],
                           [0, 1, 0, -100],
                           [0, 0, 1, -100],
                           [0, 0, 0, 1]], dtype=float)
        ed = mesh_io.ElementData.from_data_grid(sphere3_msh, X, affine)
        assert np.allclose(sphere3_msh.elements_baricenters().value[:, 0],
                           ed.value)

    def test_from_data_grid_vec(self, sphere3_msh):
        X, Y, Z = np.meshgrid(np.arange(-100, 100),
                              np.arange(-100, 100),
                              np.arange(-100, 100),
                              indexing='ij')
        V = np.stack([X, Y, Z], axis=3)
        V = V.astype(float)
        affine = np.array([[1, 0, 0, -100],
                           [0, 1, 0, -100],
                           [0, 0, 1, -100],
                           [0, 0, 0, 1]], dtype=float)
        ed = mesh_io.ElementData.from_data_grid(sphere3_msh, V, affine)
        assert np.allclose(sphere3_msh.elements_baricenters().value,
                           ed.value)

    def test_from_data_grid_slice(self, sphere3_msh):
        X, Y, Z = np.meshgrid(np.arange(-100, 100),
                              np.arange(-100, 100),
                              [0],
                              indexing='ij')
        V = np.stack([X, Y, Z], axis=3)
        V = V.astype(float)
        affine = np.array([[1, 0, 0, -100],
                           [0, 1, 0, -100],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        ed = mesh_io.ElementData.from_data_grid(
            sphere3_msh, V, affine,
            cval=np.nan, order=1
        )
        bar = sphere3_msh.elements_baricenters().value
        in_slice = (bar[:, 2] >= -.5) * (bar[:, 2] <= .5)
        assert np.allclose(bar[in_slice, :2], ed.value[in_slice, :2])
        assert np.all(np.isnan(ed.value[~in_slice]))



class TestNodeData:

    def test_write_nodedata_scalar(self, sphere3_msh):
        node_data = mesh_io.NodeData(sphere3_msh.nodes.node_coord[
                                  :, 0], 'nodedata_scalar')
        tmp = copy.deepcopy(sphere3_msh)
        tmp.nodedata.append(node_data)
        mesh_io.write_msh(tmp, 'tmp.msh')
        tmp = mesh_io.read_msh('tmp.msh')
        os.remove('tmp.msh')
        np.testing.assert_array_almost_equal(
            tmp.nodedata[0].value,
            sphere3_msh.nodes.node_coord[:, 0])

    def test_write_nodedata_vect(self, sphere3_msh):
        node_data = mesh_io.NodeData(
            sphere3_msh.nodes.node_coord, 'nodedata_vectorial')
        tmp = copy.deepcopy(sphere3_msh)
        tmp.nodedata.append(node_data)
        mesh_io.write_msh(tmp, 'tmp.msh')
        tmp = mesh_io.read_msh('tmp.msh')
        os.remove('tmp.msh')
        np.testing.assert_array_almost_equal(tmp.nodedata[0].value,
                                             sphere3_msh.nodes.node_coord)

    def test_node_data2elm_data_scalar(self, sphere3_msh):
        node_data = mesh_io.NodeData(sphere3_msh.nodes.node_coord[:, 0], mesh=sphere3_msh)
        elm_data = node_data.node_data2elm_data()
        triangles = np.where(sphere3_msh.elm.elm_type == 2)[0]
        tr_x_baricenter = np.average(sphere3_msh.nodes.node_coord[
            sphere3_msh.elm.node_number_list[triangles, :3] - 1, 0], axis=1)

        np.testing.assert_array_almost_equal(
            tr_x_baricenter,
            elm_data.value[triangles].reshape(-1))

    def test_node_data2elm_data_vectorial(self, sphere3_msh):
        node_data = mesh_io.NodeData(sphere3_msh.nodes.node_coord, mesh=sphere3_msh)
        elm_data = node_data.node_data2elm_data()
        triangles = np.where(sphere3_msh.elm.elm_type == 2)[0]
        tr_baricenter = np.average(sphere3_msh.nodes.node_coord[
            sphere3_msh.elm.node_number_list[triangles, :3] - 1, :], axis=1)

        np.testing.assert_array_almost_equal(
            tr_baricenter, elm_data.value[triangles])

    def test_gradient(self, sphere3_msh):
        node_data = mesh_io.NodeData(np.ones(sphere3_msh.nodes.nr, dtype=float),
                                  mesh=sphere3_msh)
        grad = node_data.gradient()
        assert np.allclose(grad.value, 0.)
        assert grad.field_name == 'grad_'

    def test_gradient_linear(self, sphere3_msh):
        msh = sphere3_msh.crop_mesh(elm_type=4)
        node_data = mesh_io.NodeData(np.ones(msh.nodes.nr, dtype=float),
                                  mesh=msh)
        node_data.value = msh.nodes.node_coord[:, 0]
        grad = node_data.gradient()
        assert np.allclose(grad.value, [1, 0, 0])
        node_data.value = msh.nodes.node_coord[:, 1]
        grad = node_data.gradient()
        assert np.allclose(grad.value, [0, 1, 0])
        node_data.value = msh.nodes.node_coord[:, 2]
        grad = node_data.gradient()
        assert np.allclose(grad.value, [0, 0, 1])

    def test_calc_flux(self, sphere3_msh):
        nodedata = mesh_io.NodeData(np.tile([1., 0., 0.], (sphere3_msh.nodes.nr, 1)),
                                 mesh=sphere3_msh)
        nodes = np.unique(sphere3_msh.elm.node_number_list[sphere3_msh.elm.tag1 == 1003, :3])
        flux = nodedata.calc_flux(nodes)
        assert np.isclose(flux, 0, atol=1e-1)

    def test_calc_flux_radial(self, sphere3_msh):
        nodedata = mesh_io.NodeData(sphere3_msh.nodes.node_coord, mesh=sphere3_msh)
        nodes = np.unique(sphere3_msh.elm.node_number_list[sphere3_msh.elm.tag1 == 1003, :3])
        flux = nodedata.calc_flux(nodes)
        # Divergence theorem
        assert np.isclose(flux, 85 ** 3 * 4 * np.pi, rtol=1e-2)

    def test_interpolate_scattered(self, sphere3_msh):
        msh = sphere3_msh.crop_mesh([3, 4, 5])
        nd = mesh_io.NodeData(msh.nodes.node_coord)
        nd.mesh = msh
        nd.value = nd.value[:, 0]
        interp_points = msh.elements_baricenters().value[:10]
        interp = nd.interpolate_scattered(interp_points)
        assert np.allclose(interp, interp_points[:, 0], atol=1e-1, rtol=1e-1)

    def test_interpolate_vect(self, sphere3_msh):
        msh = sphere3_msh.crop_mesh([3, 4, 5])
        nd = mesh_io.NodeData(msh.nodes.node_coord)
        nd.mesh = msh
        interp_points = msh.elements_baricenters().value[:10]
        interp = nd.interpolate_scattered(interp_points)
        assert np.allclose(interp, interp_points, atol=1e-1, rtol=1e-1)

    def test_interpolate_scattered_th(self, sphere3_msh):
        msh = sphere3_msh.crop_mesh([3, 4, 5])
        nd = mesh_io.NodeData(msh.nodes.node_coord)
        nd.mesh = msh
        interp_points = msh.elements_baricenters().value[:10]
        th_indices = msh.elm.elm_number[np.isin(msh.elm.tag1, [3, 4, 5])]
        interp = nd.interpolate_scattered(
            interp_points,  th_indices=th_indices)
        assert np.allclose(interp, interp_points, atol=1e-1, rtol=1e-1)

    def test_norm(self, sphere3_msh):
        nd = mesh_io.NodeData(sphere3_msh.nodes.node_coord)
        assert np.allclose(nd.norm().value, np.linalg.norm(sphere3_msh.nodes.node_coord, axis=1))

        nd = mesh_io.NodeData(sphere3_msh.nodes.node_coord[:, 0])
        assert np.allclose(nd.norm().value, np.abs(sphere3_msh.nodes.node_coord[:, 0]))


    def test_normal(self, sphere3_msh):
        nd = mesh_io.NodeData(sphere3_msh.nodes.node_coord, mesh=sphere3_msh)
        outer_surface = sphere3_msh.elm.node_number_list[sphere3_msh.elm.tag1==1005, :3]
        outer_nodes = np.unique(outer_surface)
        normal = nd.normal()
        assert np.allclose(normal[outer_nodes], 95, rtol=1e-3)


    def test_angles(self, sphere3_msh):
        nd = mesh_io.NodeData(sphere3_msh.nodes.node_coord, mesh=sphere3_msh)
        outer_surface = sphere3_msh.elm.node_number_list[sphere3_msh.elm.tag1==1005, :3]
        outer_nodes = np.unique(outer_surface)
        angle = nd.angle()
        assert np.allclose(angle[outer_nodes], 0, atol=1e-1)


    def test_tangent(self, sphere3_msh):
        nd = mesh_io.NodeData(sphere3_msh.nodes.node_coord, mesh=sphere3_msh)
        outer_surface = sphere3_msh.elm.node_number_list[sphere3_msh.elm.tag1==1005, :3]
        outer_nodes = np.unique(outer_surface)
        tangent = nd.tangent()
        assert np.allclose(tangent[outer_nodes] / nd.norm()[outer_nodes], 0, atol=1e-1)

    def test_from_data_grid(self, sphere3_msh):
        X, Y, Z = np.meshgrid(np.arange(-100, 100),
                              np.arange(-100, 100),
                              np.arange(-100, 100),
                              indexing='ij')
        X = X.astype(float)
        affine = np.array([[1, 0, 0, -100],
                           [0, 1, 0, -100],
                           [0, 0, 1, -100],
                           [0, 0, 0, 1]], dtype=float)
        nd = mesh_io.NodeData.from_data_grid(sphere3_msh, X, affine)
        assert np.allclose(sphere3_msh.nodes[:, 0], nd.value)

    def test_from_data_grid_slice(self, sphere3_msh):
        X, Y, Z = np.meshgrid(np.arange(-100, 100),
                              np.arange(-100, 100),
                              [0],
                              indexing='ij')
        V = np.stack([X, Y, Z], axis=3)
        V = V.astype(float)
        affine = np.array([[1, 0, 0, -100],
                           [0, 1, 0, -100],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        ed = mesh_io.NodeData.from_data_grid(
            sphere3_msh, V, affine,
            cval=np.nan, order=1
        )
        in_slice = (sphere3_msh.nodes[:, 2] >= -.5) * (sphere3_msh.nodes[:, 2] <= .5)
        assert np.allclose(sphere3_msh.nodes[in_slice, :2], ed.value[in_slice, :2])
        assert np.all(np.isnan(ed.value[~in_slice]))




class TestReadRes:

    def test_read_res_ascii(self):
        with open('tmp.res', 'bw') as f:
            f.write('\n'.join(['$ResFormat /* GetDP 2.8.0, ascii */',
                               '1.1 0',
                               '$EndResFormat',
                               '$Solution  /* DofData #0 */',
                               '0 0 0 0',
                               '-1.2 0',
                               '-2 0',
                               '3.0 0',
                               '$EndSolution']).encode('ascii'))

        v = mesh_io.read_res_file('tmp.res')
        os.remove('tmp.res')

        np.testing.assert_allclose([-1.2, -2, 3.0], v)

    def test_read_res_bin(self):
        with open('tmp.res', 'bw') as f:
            f.write(b'$ResFormat /* GetDP 2.8.0, ascii */\n')
            f.write(b'1.1 1\n')
            f.write(b'$EndResFormat\n')
            f.write(b'$Solution\n')
            f.write(b'0 0 0 0\n')
            f.write(np.array([-1.2, 0, -2, 0, 3.0, 0],dtype='float64').tobytes())
            f.write(b'\n')
            f.write(b'$EndSolution\n')

        v = mesh_io.read_res_file('tmp.res')
        os.remove('tmp.res')

        np.testing.assert_allclose([-1.2, -2, 3.0], v)

class TestVTK:
    import importlib
    @pytest.mark.skipif(
        not importlib.util.find_spec("pyvista"), reason="requires the pyvista library"
    )
    def test_read_write(self, tmp_path):
        nodes = np.array([[1.2, 3.4, 1.2], [2.3, 5.6, 1.2], [3.1, 6.7, 9.2], [9.1, 2.5, 2.3], [23.1, 3.4, 5.3]])
        triangles = np.array([[1,2,3], [2,4,5]])
        tetrahedra = np.array([[1,2,3,4], [3, 4, 5, 1]])

        msh = mesh_io.Msh(mesh_io.Nodes(nodes), mesh_io.Elements(triangles, tetrahedra))

        msh.add_element_field(np.arange(msh.elm.nr) + 99, 'elm_test')
        msh.add_node_field(np.arange(msh.nodes.nr)+ 999, 'node_test')

        msh.elm.tag1 = np.arange(msh.elm.nr)
        msh.elm.tag2 = np.arange(msh.elm.nr)

        path = os.path.join(tmp_path, 'test.vtk')
        mesh_io.write_vtk(msh, path)
        vtk_msh = mesh_io.read_vtk(path)

        assert msh.nodes.nr == vtk_msh.nodes.nr
        assert msh.elm.nr == vtk_msh.elm.nr
        assert len(msh.nodedata) == len(vtk_msh.nodedata)
        assert len(msh.elmdata) == len(vtk_msh.elmdata)

        np.testing.assert_array_equal(msh.elm.node_number_list, vtk_msh.elm.node_number_list)
        np.testing.assert_allclose(msh.nodes.node_coord, vtk_msh.nodes.node_coord)
        np.testing.assert_array_equal(msh.elm.tag1, vtk_msh.elm.tag1)
        np.testing.assert_array_equal(msh.elm.tag2, vtk_msh.elm.tag2)

        for msh_elm_data, vtk_msh_elm_data in zip(msh.elmdata, vtk_msh.elmdata):
            assert msh_elm_data.field_name == vtk_msh_elm_data.field_name
            np.testing.assert_array_equal(msh_elm_data.value, vtk_msh_elm_data.value)

        for msh_node_data, vtk_node_elm_data in zip(msh.nodedata, vtk_msh.nodedata):
            assert msh_node_data.field_name == vtk_node_elm_data.field_name
            np.testing.assert_array_equal(msh_node_data.value, vtk_node_elm_data.value)

class TestWriteGeo:
    def test_write_spheres_no_values(self):
        positions = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        mesh_io.write_geo_spheres(positions, 'tst.geo')
        with open('tst.geo', 'br') as f:
            assert f.read().decode('ascii') == (
                'View""{\n'
                'SP(1.0, 0.0, 0.0){0.0};\n'
                'SP(0.0, 1.0, 0.0){0.0};\n'
                '};\n')
        os.remove('tst.geo')
    def test_write_sphere_field_values(self):
        positions = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        fields = [1, -1]
        mesh_io.write_geo_spheres(positions, 'tst.geo', values=fields)
        with open('tst.geo') as f:
            assert f.read() == ('View""{\n'
                                'SP(1.0, 0.0, 0.0){1};\n'
                                'SP(0.0, 1.0, 0.0){-1};\n'
                                '};\n')
        os.remove('tst.geo')
    def test_write_sphere_field_name(self):
        positions = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        mesh_io.write_geo_spheres(positions, 'tst.geo', name="Foo")
        with open('tst.geo') as f:
            assert f.read() == ('View"Foo"{\n'
                                'SP(1.0, 0.0, 0.0){0.0};\n'
                                'SP(0.0, 1.0, 0.0){0.0};\n'
                                '};\n')
        os.remove('tst.geo')


class TestSurfaceIO:
    @pytest.mark.parametrize('binary', [True, False])
    def test_stl_io(self, binary, sphere3_msh):
        surf = sphere3_msh.crop_mesh(1005)
        mesh_io.write_stl(surf, 'tst.stl', binary)
        surf_read = mesh_io.read_stl('tst.stl')
        os.remove('tst.stl')
        # STL format changes order
        assert np.allclose(surf.nodes[surf.elm[:, :3]],
                           surf_read.nodes[surf_read.elm[:, :3]])

    def test_off_io(self, sphere3_msh):
        surf = sphere3_msh.crop_mesh(1005)
        mesh_io.write_off(surf, 'tst.off')
        surf_read = mesh_io.read_off('tst.off')
        os.remove('tst.off')
        assert np.allclose(surf.nodes[surf.elm[:, :3]],
                           surf_read.nodes[surf_read.elm[:, :3]])

    def test_gifti_io(self, sphere3_msh):
        surf = sphere3_msh.crop_mesh(1005)
        mesh_io.write_gifti_surface(surf, 'tst.gii')
        surf_read = mesh_io.read_gifti_surface('tst.gii')
        os.remove('tst.gii')
        assert np.allclose(surf.nodes[surf.elm[:, :3]],
                           surf_read.nodes[surf_read.elm[:, :3]])



class TestHashing:
    def test_collisions(self):
        np.random.seed(0)
        array = np.random.randint(0, 1e6, size=(int(1e7), 3), dtype=np.int32)
        array = np.unique(array, axis=0)
        hash_ = mesh_io._hash_rows(np.array(array))
        _, count = np.unique(hash_, return_counts=True)
        assert np.all(count == 1)

class TestNodeRasterization:
    def test_interpolate_to_grid_max(self, atlas_itk_msh):
        l, p = atlas_itk_msh.nodedata[0].interpolate_to_grid_max(
            np.max(atlas_itk_msh.nodes.node_coord,axis=0).astype(int),
            np.identity(4), parallel=False)
        assert np.allclose(np.bincount(l.ravel()),(49984, 9335))
        assert np.allclose(np.bincount(p.ravel().astype(int)),(5652, 53667))

        lp, pp = atlas_itk_msh.nodedata[0].interpolate_to_grid_max(
            np.max(atlas_itk_msh.nodes.node_coord,axis=0).astype(int),
            np.identity(4), parallel=True)
        assert(np.allclose(l,lp))
        assert(np.allclose(p,pp))

        lp, pp = atlas_itk_msh.nodedata[0].interpolate_to_grid_max(
            np.max(atlas_itk_msh.nodes.node_coord,axis=0).astype(int), compartments=[[1],[0]],
            affine=np.identity(4), parallel=True)
        assert(np.all(l!=lp))
        assert(np.allclose(p,pp))

class TestAABBTree:
    def test_AABBTree(self, sphere3_msh):
        tree = sphere3_msh.get_AABBTree()

        insideidx = tree.points_inside(np.array(((0,0,0),(50,50,50),(10,50,50))))
        assert(insideidx==[0,2])
        assert(tree.any_point_inside(np.array(((0,0,0),(50,30,25)))))
        assert(tree.any_point_inside(np.array((50,30,25))==False))
        # tree.__del__()
        del tree

class TestGetMinDistanceOnGrid:
    @pytest.mark.parametrize("tag", [1003, 1004, 1005])
    @pytest.mark.parametrize("resolution", [1, 2])
    @pytest.mark.parametrize("offset", [-3.4, -1.2, 0, 1, 4.567])
    def test_spheres_internal_distance(self, sphere3_msh: mesh_io.Msh, tag, resolution, offset):
        surface = sphere3_msh.crop_mesh([tag])
        min_distance_on_grid, grid, M, AABBTree = surface.get_min_distance_on_grid(resolution=resolution, distance_offset=offset)

        radius = np.max(surface.nodes.node_coord[:, 0]) + offset
        max_internal_distance = np.min(grid) * -1 
        assert radius - resolution < max_internal_distance and radius + resolution > max_internal_distance


    def test_distance_function_on_sphere(self, sphere3_msh: mesh_io.Msh,):
        surface = sphere3_msh.crop_mesh([1005])
        min_distance_on_grid, grid, M, AABBTree = surface.get_min_distance_on_grid(resolution=1)

        radius = np.max(surface.nodes.node_coord[:, 0]) 
        for offset in range(200):
            distance = min_distance_on_grid(np.array([[offset, 0, 0]]))[0] * -1
            assert radius - offset - 1.5 < distance and radius - offset + 1.5 > distance


