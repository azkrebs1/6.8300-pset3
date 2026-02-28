#!/usr/bin/env python3
"""
Local test suite for PSET 3 - Structure from Motion Pipeline
6.8300 - Advances in Computer Vision

Run from the student_pset3 directory:
    python student_autograder.py

These tests use DIFFERENT values than the Gradescope autograder.
"""

import sys
import unittest
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np
import nbformat

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def extract_functions_from_notebook(notebook_path: str, namespace: dict) -> dict:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        code = cell.source
        if isinstance(code, list):
            code = ''.join(code)
        if code.strip().startswith('%') or code.strip().startswith('!'):
            continue
        if 'import env' in code or 'from env' in code or 'import src.' in code or 'from src.' in code:
            continue
        if 'plt.show()' in code and 'def ' not in code:
            continue
        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                exec(code, namespace)
        except Exception:
            pass
    return namespace


def find_notebook(pattern: str) -> str:
    search_path = Path(".")
    notebooks = list(search_path.glob(f"*{pattern}*.ipynb"))
    if not notebooks:
        notebooks = list(search_path.glob(f"**/*{pattern}*.ipynb"))
    if notebooks:
        student_nbs = [nb for nb in notebooks if "solution" not in nb.name.lower()]
        if student_nbs:
            return str(student_nbs[0])
        return str(notebooks[0])
    return None


NAMESPACE = {}


def load_functions():
    global NAMESPACE
    if NAMESPACE:
        return
    if not HAS_CV2:
        return
    notebook_path = find_notebook("sfm_pipeline")
    if notebook_path:
        print(f"Loading from {notebook_path}")
        namespace = {'np': np, 'numpy': np}
        try:
            import cv2
            namespace['cv2'] = cv2
        except ImportError:
            pass
        try:
            from PIL import Image
            namespace['Image'] = Image
        except ImportError:
            pass
        NAMESPACE = extract_functions_from_notebook(notebook_path, namespace)
    else:
        print("No notebook found")


def get_func(name):
    return NAMESPACE.get(name)


def get_data_dir():
    """Data directory for reference tests (expected*.npy and inputs). Same layout as env.py.
    Tries script_dir/data first (zip handout: data inside student_pset3), then parent/data (repo layout)."""
    script_dir = Path(__file__).resolve().parent
    in_dir = script_dir / "data"
    if in_dir.exists():
        return in_dir
    return script_dir.parent / "data"


class TestPart1ContourImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_binarize(self):
        ContourImage = get_func('ContourImage')
        if ContourImage is None:
            self.skipTest("ContourImage not implemented")
        from PIL import Image
        np.random.seed(100)
        arr = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        img = Image.fromarray(arr)
        ci = ContourImage(img)
        ci.binarize(threshold=128)
        self.assertIsNotNone(ci.binarized_image)
        self.assertEqual(ci.binarized_image.shape, (10, 10))
        self.assertTrue(np.isin(ci.binarized_image, [0, 1]).all())

    def test_fill_border(self):
        ContourImage = get_func('ContourImage')
        if ContourImage is None:
            self.skipTest("ContourImage not implemented")
        from PIL import Image
        img = Image.fromarray(np.ones((5, 5), dtype=np.uint8) * 255)
        ci = ContourImage(img)
        ci.binarize(threshold=128)
        ci.fill_border()
        self.assertEqual(ci.binarized_image[0, :].sum(), 0)
        self.assertEqual(ci.binarized_image[-1, :].sum(), 0)

    def test_find_contours(self):
        """Student implements is_boundary_neighbor; find_contours uses it and returns list of (row, col)."""
        find_contours = get_func('find_contours')
        if find_contours is None:
            self.skipTest("find_contours not implemented")
        binary = np.zeros((6, 6), dtype=np.uint8)
        binary[2:5, 2:5] = 1
        contours = find_contours(binary, foreground=1)
        self.assertIsInstance(contours, list, "find_contours should return a list")
        self.assertGreater(len(contours), 0,
                          "find_contours should return boundary pixels; check your boundary condition (neighbor out-of-bounds or not foreground)")
        self.assertEqual(len(contours[0]), 2,
                         f"find_contours should return a list of (row, col) pairs; got element of length {len(contours[0])}")


class TestPart2Calibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_get_3D_object_points(self):
        func = get_func('get_3D_object_points')
        if func is None:
            self.skipTest("get_3D_object_points not implemented")
        pts = func((4, 3))
        self.assertEqual(pts.shape, (12, 3))
        np.testing.assert_array_almost_equal(pts[0], [0, 0, 0])
        np.testing.assert_array_almost_equal(pts[-1], [3, 2, 0])
        self.assertTrue(np.all(pts[:, 2] == 0))

    def test_undistort_image(self):
        func = get_func('undistort_image')
        if func is None:
            self.skipTest("undistort_image not implemented")
        K = np.array([[80, 0, 40], [0, 80, 40], [0, 0, 1]], dtype=np.float64)
        dist = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        np.random.seed(101)
        img = np.random.randint(0, 256, (15, 15, 3), dtype=np.uint8)
        out = func(img, K, dist)
        self.assertEqual(out.shape, img.shape)

    def test_p2_reference_camera_matrix(self):
        """Reference test: compare to expected*.npy (same as 2025-solutions). Uses chessboard (16,9)."""
        data_dir = get_data_dir()
        chess_path = data_dir / "p1_edge_identification" / "chessboard.png"
        exp_K = data_dir / "p2_calibrate_camera" / "expected_camera_matrix.npy"
        exp_d = data_dir / "p2_calibrate_camera" / "expected_dist_coeffs.npy"
        if not chess_path.exists() or not exp_K.exists() or not exp_d.exists():
            self.skipTest("Reference data not found (expected*.npy and chessboard)")
        load_image = get_func("load_image")
        load_grayscale = get_func("load_grayscale_image")
        find_corners = get_func("find_chessboard_corners")
        refine = get_func("refine_corners")
        get_obj = get_func("get_3D_object_points")
        calibrate = get_func("calibrate_camera")
        if any(x is None for x in (load_image, load_grayscale, find_corners, refine, get_obj, calibrate)):
            self.skipTest("Required P2 functions not implemented")
        expected_K = np.load(exp_K)
        expected_d = np.load(exp_d)
        image = load_image(chess_path)
        gray = load_grayscale(image)
        chessboard_size = (16, 9)  # match 2025-solutions
        corners = find_corners(gray, chessboard_size)
        corners = refine(gray, corners)
        object_points = get_obj(chessboard_size)
        camera_matrix, dist_coeffs = calibrate(object_points, corners, gray.shape[::-1])
        np.testing.assert_allclose(camera_matrix, expected_K, atol=1e-2)
        np.testing.assert_allclose(dist_coeffs, expected_d, atol=1e-2)


class TestPart3Fundamental(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_lstsq_eight_point_alg(self):
        func = get_func('lstsq_eight_point_alg')
        if func is None:
            self.skipTest("lstsq_eight_point_alg not implemented")
        np.random.seed(102)
        n = 10
        points1 = np.random.randn(n, 3)
        points1[:, 2] = 1
        F_true = np.random.randn(3, 3)
        F_true = F_true - F_true.mean()
        points2 = (F_true @ points1.T).T
        points2 = points2 / points2[:, 2:3]
        F = func(points1, points2)
        self.assertEqual(F.shape, (3, 3))
        rank = np.linalg.matrix_rank(F)
        self.assertAlmostEqual(rank, 2, msg=f"F must have rank 2; got rank {rank}.")

    def test_normalized_eight_point_alg(self):
        func = get_func('normalized_eight_point_alg')
        if func is None:
            self.skipTest("normalized_eight_point_alg not implemented")
        np.random.seed(103)
        points1 = np.random.rand(10, 3) * 50
        points1[:, 2] = 1
        points2 = points1 + np.random.randn(10, 3)
        points2[:, 2] = 1
        F = func(points1, points2)
        self.assertEqual(F.shape, (3, 3))

    def test_compute_epipolar_lines(self):
        func = get_func('compute_epipolar_lines')
        if func is None:
            self.skipTest("compute_epipolar_lines not implemented")
        F = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        points = np.array([[0, 0, 1], [1, 1, 1]])
        lines = func(points, F)
        self.assertEqual(len(lines), 2)
        self.assertEqual(len(lines[0]), 2,
                         f"compute_epipolar_lines should return a list of (m, b) pairs; got element of length {len(lines[0])}")

    def test_p3_reference_F_and_dist(self):
        """Reference test: compare F and distances to expected*.npy (same as 2025-solutions)."""
        data_dir = get_data_dir()
        p3 = data_dir / "p3_fundamental_matrix"
        pts1_path = p3 / "pts_1.txt"
        pts2_path = p3 / "pts_2.txt"
        for p in (p3 / "expected_F_LLS.npy", p3 / "expected_dist_LLS.npy",
                  p3 / "expected_F_normalized.npy", p3 / "expected_dist_normalized.npy"):
            if not p.exists():
                self.skipTest("Reference data not found (expected*.npy)")
        load_points = get_func("load_points")
        lstsq = get_func("lstsq_eight_point_alg")
        norm_f = get_func("normalized_eight_point_alg")
        dist_epi = get_func("compute_distance_to_epipolar_lines")
        if any(x is None for x in (load_points, lstsq, norm_f, dist_epi)):
            self.skipTest("Required P3 functions not implemented")
        pts1 = load_points(pts1_path)
        pts2 = load_points(pts2_path)
        expected_F_lls = np.load(p3 / "expected_F_LLS.npy")
        expected_dist_lls = np.load(p3 / "expected_dist_LLS.npy")
        expected_F_norm = np.load(p3 / "expected_F_normalized.npy")
        expected_dist_norm = np.load(p3 / "expected_dist_normalized.npy")
        F_lls = lstsq(pts1, pts2)
        np.testing.assert_allclose(F_lls, expected_F_lls, atol=1e-2)
        d1_lls = dist_epi(pts1, pts2, F_lls)
        d2_lls = dist_epi(pts2, pts1, F_lls.T)
        np.testing.assert_allclose(d1_lls, expected_dist_lls[0], atol=1e-2)
        np.testing.assert_allclose(d2_lls, expected_dist_lls[1], atol=1e-2)
        F_norm = norm_f(pts1, pts2)
        np.testing.assert_allclose(F_norm, expected_F_norm, atol=1e-2)
        d1_n = dist_epi(pts1, pts2, F_norm)
        d2_n = dist_epi(pts2, pts1, F_norm.T)
        np.testing.assert_allclose(d1_n, expected_dist_norm[0], atol=1e-2)
        np.testing.assert_allclose(d2_n, expected_dist_norm[1], atol=1e-2)


class TestPart4Rectification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_compute_epipole(self):
        func = get_func('compute_epipole')
        if func is None:
            self.skipTest("compute_epipole not implemented")
        np.random.seed(104)
        n = 8
        p1 = np.random.rand(n, 3)
        p1[:, 2] = 1
        F = np.random.randn(3, 3)
        e = func(p1, p1, F)
        self.assertEqual(len(e), 3)
        self.assertAlmostEqual(e[2], 1.0)

    def test_compute_rectified_image(self):
        func = get_func('compute_rectified_image')
        if func is None:
            self.skipTest("compute_rectified_image not implemented")
        np.random.seed(105)
        img = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        H = np.eye(3)
        out, offsets = func(img, H)
        self.assertIsInstance(offsets, (tuple, list))
        self.assertEqual(out.ndim, 3)

    def test_find_matches(self):
        func = get_func('find_matches')
        if func is None:
            self.skipTest("find_matches not implemented")
        np.random.seed(106)
        img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
        kp1, kp2, matches = func(img, img)
        self.assertIsInstance(matches, list)

    def test_p4_reference_epipole_homographies(self):
        """Reference test: compare e1, e2, H1, H2 to expected*.npy (same as 2025-solutions)."""
        data_dir = get_data_dir()
        p3 = data_dir / "p3_fundamental_matrix"
        p4 = data_dir / "p4_image_rectification"
        for name in ("expected_e1.npy", "expected_e2.npy", "expected_H1.npy", "expected_H2.npy"):
            if not (p4 / name).exists():
                self.skipTest("Reference data not found (expected*.npy)")
        load_points = get_func("load_points")
        load_image = get_func("load_image")
        norm_f = get_func("normalized_eight_point_alg")
        epipole = get_func("compute_epipole")
        homographies = get_func("compute_matching_homographies")
        if any(x is None for x in (load_points, load_image, norm_f, epipole, homographies)):
            self.skipTest("Required P4 functions not implemented")
        pts1 = load_points(p3 / "pts_1.txt")
        pts2 = load_points(p3 / "pts_2.txt")
        im1 = load_image(p3 / "const_im1.png")
        im2 = load_image(p3 / "const_im2.png")
        F = norm_f(pts1, pts2)
        e1 = epipole(pts1, pts2, F)
        e2 = epipole(pts2, pts1, F.T)
        expected_e1 = np.load(p4 / "expected_e1.npy")
        expected_e2 = np.load(p4 / "expected_e2.npy")
        # Epipole is unique up to sign (null space of F); accept either e or -e
        for e, exp in [(e1, expected_e1), (e2, expected_e2)]:
            match = np.allclose(e, exp, rtol=1e-2) or np.allclose(e, -exp, rtol=1e-2)
            self.assertTrue(match, msg=f"Epipole {e} should match expected {exp} up to sign")
        H1, H2 = homographies(e2, F, im2, pts1, pts2)
        # Allow tolerance for homographies (implementation details can differ; ~6% relative seen in solutions)
        np.testing.assert_allclose(H1, np.load(p4 / "expected_H1.npy"), rtol=7e-2)
        np.testing.assert_allclose(H2, np.load(p4 / "expected_H2.npy"), rtol=7e-2)


class TestPart53DReconstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_compute_essential_matrix(self):
        func = get_func('compute_essential_matrix')
        if func is None:
            self.skipTest("compute_essential_matrix not implemented")
        K = np.eye(3)
        F = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
        E = func(K, F)
        self.assertEqual(E.shape, (3, 3))

    def test_estimate_initial_RT(self):
        func = get_func('estimate_initial_RT')
        if func is None:
            self.skipTest("estimate_initial_RT not implemented")
        E = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float64)
        R_cand, T_cand = func(E)
        self.assertEqual(len(R_cand), 2, "estimate_initial_RT should return (candidate_Rs, candidate_ts) with len(candidate_Rs)==2")
        self.assertEqual(len(T_cand), 2, "estimate_initial_RT should return (candidate_Rs, candidate_ts) with len(candidate_ts)==2")

    def test_get_identity_projection_matrix(self):
        func = get_func('get_identity_projection_matrix')
        if func is None:
            self.skipTest("get_identity_projection_matrix not implemented")
        K = np.eye(3)
        P = func(K)
        self.assertEqual(P.shape, (3, 4))
        np.testing.assert_array_almost_equal(P[:, :3], np.eye(3))

    def test_get_local_projection_matrix(self):
        func = get_func('get_local_projection_matrix')
        if func is None:
            self.skipTest("get_local_projection_matrix not implemented")
        K = np.eye(3)
        R = np.eye(3)
        T = np.zeros(3)
        P = func(K, R, T)
        self.assertEqual(P.shape, (3, 4))

    def test_p5_reference_R_T(self):
        """Reference test: compare R, T to expected*.npy (same as 2025-solutions). Uses P5 chessboard (16,10)."""
        np.random.seed(42)  # Reproducible pipeline for RANSAC/SIFT matching
        data_dir = get_data_dir()
        p5 = data_dir / "p5_3D_reconstruction"
        chess_path = p5 / "chessboard.png"
        im1_path = p5 / "raw_images" / "object_0.png"
        im2_path = p5 / "raw_images" / "object_1.png"
        for p in (chess_path, im1_path, im2_path, p5 / "expected_R.npy", p5 / "expected_T.npy"):
            if not p.exists():
                self.skipTest("Reference data not found (expected R/T, P5 chessboard.png, or raw images)")
        load_image = get_func("load_image")
        load_grayscale = get_func("load_grayscale_image")
        find_corners = get_func("find_chessboard_corners")
        refine = get_func("refine_corners")
        get_obj = get_func("get_3D_object_points")
        calibrate = get_func("calibrate_camera")
        find_matches = get_func("find_matches")
        recover_F = get_func("recover_fundamental_matrix")
        get_inliers = get_func("get_inliers")
        compute_E = get_func("compute_essential_matrix")
        estimate_RT = get_func("estimate_initial_RT")
        find_best_RT = get_func("find_best_RT")
        if any(x is None for x in (load_image, load_grayscale, find_corners, refine, get_obj, calibrate,
                                   find_matches, recover_F, get_inliers, compute_E, estimate_RT, find_best_RT)):
            self.skipTest("Required P5 functions not implemented")
        expected_R = np.load(p5 / "expected_R.npy")
        expected_T = np.squeeze(np.load(p5 / "expected_T.npy"))
        chessboard_size = (16, 10)  # match 2025-solutions (p5_3D_reconstruction.py)
        # Calibrate from dedicated chessboard.png (same as 2025: calibrate_camera_from_chessboard(env.p5.chessboard))
        chess_img = load_image(chess_path)
        gray_chess = load_grayscale(chess_img)
        try:
            corners_chess = find_corners(gray_chess, chessboard_size)
        except Exception as e:
            self.fail(
                f"P5 calibration requires chessboard (16,10) detectable in data/p5_3D_reconstruction/chessboard.png; "
                f"detection failed: {e}"
            )
        corners_chess = refine(gray_chess, corners_chess)
        object_points = get_obj(chessboard_size)
        camera_matrix, _ = calibrate(object_points, corners_chess, gray_chess.shape[::-1])
        # Two-view pipeline from object_0 and object_1 (no chessboard required in these images)
        im1 = load_image(im1_path)
        im2 = load_image(im2_path)
        kp1, kp2, good_matches = find_matches(im1, im2)
        F, mask, pts1, pts2 = recover_F(kp1, kp2, good_matches)
        inlier_pts1, inlier_pts2 = get_inliers(mask, pts1, pts2)
        E = compute_E(camera_matrix, F)
        R_cand, T_cand = estimate_RT(E)
        R, T = find_best_RT(R_cand, T_cand, inlier_pts1, inlier_pts2)
        T = np.squeeze(T)
        # Reference test tolerance: P5 R/T allow some variation due to RANSAC and scale ambiguity.
        np.testing.assert_allclose(R, expected_R, atol=0.08)
        np.testing.assert_allclose(T, expected_T, atol=0.15)


def create_suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    if HAS_CV2:
        suite.addTests(loader.loadTestsFromTestCase(TestPart1ContourImage))
        suite.addTests(loader.loadTestsFromTestCase(TestPart2Calibration))
        suite.addTests(loader.loadTestsFromTestCase(TestPart3Fundamental))
        suite.addTests(loader.loadTestsFromTestCase(TestPart4Rectification))
        suite.addTests(loader.loadTestsFromTestCase(TestPart53DReconstruction))
    return suite


def main():
    print("=" * 70)
    print("Local Test Suite - PSET 3 Structure from Motion")
    print("6.8300 - Advances in Computer Vision")
    print("=" * 70)
    print("\nNote: These tests use DIFFERENT values than Gradescope.")
    print("Passing here means your implementation is likely correct!\n")
    load_functions()
    suite = create_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("\n" + "=" * 70)
    print(f"TOTAL: {result.testsRun} tests, {len(result.failures)} failures, "
          f"{len(result.errors)} errors, {len(result.skipped)} skipped")
    if result.failures:
        print("\n--- FAILURE DETAILS ---")
        for test, traceback in result.failures:
            print(f"FAILED: {test}")
            print(traceback)
    if result.errors:
        print("\n--- ERROR DETAILS ---")
        for test, traceback in result.errors:
            print(f"ERROR: {test}")
            print(traceback)
    print("=" * 70)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
