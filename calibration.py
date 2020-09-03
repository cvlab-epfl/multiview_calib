import os
import numpy as np
import imageio
import itertools 
import random
import logging
import cv2
import networkx as nx
from networkx.algorithms.tree.mst import maximum_spanning_edges

from . import utils 
from .singleview_geometry import undistort_points, project_points
from .twoview_geometry import (compute_relative_pose, residual_error,
                               sampson_distance, draw_epilines, triangulate, fundamental_from_relative_pose)
from .point_set_registration import (estimate_scale_point_sets, procrustes_registration, 
                                     point_set_registration, apply_rigid_transform)
from .utils import colors as view_colors

logger = logging.getLogger(__name__)

def verify_view_tree(view_tree):
    G = nx.DiGraph()
    for view1, view2 in view_tree:
        G.add_edge(view1, view2)

    try:
        nx.algorithms.cycles.find_cycle(G)
        no_cycles_found = False
    except:
        no_cycles_found = True
        
    is_connected = nx.number_connected_components(G.to_undirected())==1
    
    is_valid = no_cycles_found and is_connected
        
    return is_valid

def verify_landmarks(landmarks):
    for view,val in landmarks.items():
        if 'ids' not in val:
            return False, "landmarks file must contain the 'ids'"
        if 'landmarks' not in val:
            return False, "landmarks file must contain the 'landmarks'"
        unique = len(set(val['ids']))==len(val['ids'])
        if not unique:
            return False, "landmarks file contains duplicate IDs!" 
        number = len(val['landmarks'])==len(val['ids'])
        if not number:
            return False, "the number of IDs and landmarks in landmark file is not the same!" 
    return True, ""

def _common_landmarks(landmarks1, landmarks2, ids1, ids2):
    
    ids_common = list(set(ids1).intersection(ids2))
            
    pts1 = np.vstack([landmarks1[ids1.index(i)] for i in ids_common])
    pts2 = np.vstack([landmarks2[ids2.index(i)] for i in ids_common])
    
    return pts1, pts2, ids_common

def visualise_epilines(view_tree, relative_poses, intrinsics, landmarks, filenames,
                       output_path="output/relative_poses/"):
    
    for view1, view2 in view_tree: 
        fnames = utils.json_read(filenames)

        img1 = imageio.imread(fnames[view1])
        img2 = imageio.imread(fnames[view2])
               
        K1 = np.float64(intrinsics[view1]['K'])
        K2 = np.float64(intrinsics[view2]['K'])
        dist1 = np.float64(intrinsics[view1]['dist'])
        dist2 = np.float64(intrinsics[view2]['dist'])

        img1_undist = cv2.undistort(img1.copy(), K1, dist1, None, K1)
        img2_undist = cv2.undistort(img2.copy(), K2, dist2, None, K2)
        
        pts1, pts2, _ = _common_landmarks(landmarks[view1]["landmarks"],
                                          landmarks[view2]["landmarks"],
                                          landmarks[view1]["ids"],
                                          landmarks[view2]["ids"])
        
        pts1_undist = undistort_points(pts1, K1, dist1)
        pts2_undist = undistort_points(pts2, K2, dist2)
        
        if 'F' in relative_poses[(view1, view2)]:
            F = np.array(relative_poses[(view1, view2)]['F'])
        else:
            Rd = np.array(relative_poses[(view1, view2)]['Rd'])
            td = np.array(relative_poses[(view1, view2)]['td'])

            F = fundamental_from_relative_pose(Rd, td, K1, K2)

        idx = np.arange(pts1_undist.shape[0])
        np.random.shuffle(idx)
        img1_, img2_ = draw_epilines(img1_undist, img2_undist, pts1_undist[idx[:50]], pts2_undist[idx[:50]],
                                     F, None, linewidth=2, markersize=20)

        utils.mkdir(output_path)

        hmin = np.minimum(img1_.shape[0], img2_.shape[0])
        imageio.imsave(os.path.join(output_path, "{}_{}.jpg".format(view1, view2)), np.hstack([img1_[:hmin],img2_[:hmin]]))    

def _print_relative_pose_info(F, Rd, td, pts1_undist, pts2_undist, verbose=2, print_prefix=''):
    
    if verbose>1:
        logging.info("{}\tFundamental matrix:\n{}\t\t{}\n{}\t\t{}\n{}\t\t{}".format(print_prefix, 
                                                                              print_prefix, F[0], 
                                                                              print_prefix, F[1], 
                                                                              print_prefix, F[2]))
        logging.info("{}\tRight camera position:\n{}\t\t{}".format(print_prefix, 
                                                             print_prefix, utils.invert_Rt(Rd, td)[1].ravel()))
    if verbose>0:
        logging.info("{}\tResidual error: {}".format(print_prefix, residual_error(pts1_undist, pts2_undist, F)[0]))
        logging.info("{}\tSampson distance: {}".format(print_prefix, sampson_distance(pts1_undist, pts2_undist, F)[0]))    
        
def compute_relative_poses(view_tree, intrinsics, landmarks, 
                            method='8point', th=20, verbose=2, print_prefix=''):
    
    relative_poses = {}
    for view1, view2 in view_tree:       

        pts1, pts2, ids_common = _common_landmarks(landmarks[view1]["landmarks"],
                                                   landmarks[view2]["landmarks"],
                                                   landmarks[view1]["ids"],
                                                   landmarks[view2]["ids"])
        
        K1 = np.float64(intrinsics[view1]['K'])
        K2 = np.float64(intrinsics[view2]['K'])
        dist1 = np.float64(intrinsics[view1]['dist'])
        dist2 = np.float64(intrinsics[view2]['dist'])         

        Rd, td, F, pts1_undist, pts2_undist, tri, mask = compute_relative_pose(pts1, pts2,
                                                                               K1=K1, dist1=dist1,
                                                                               K2=K2, dist2=dist2, 
                                                                               method=method, th=th)
        
        if verbose>0:
            logging.info("{}Computing relative pose of pair {}:".format(print_prefix, [view1, view2]))
            logging.info("{}\t{} out of {} points considered outliers.".format(print_prefix, sum(~mask), len(pts1_undist)))
        _print_relative_pose_info(F, Rd, td, pts1_undist[mask], pts2_undist[mask], verbose, print_prefix)

        relative_poses[(view1, view2)] = {"F":F.tolist(), "Rd":Rd.tolist(), "td":td.tolist(), 
                                          "triang_points":tri.tolist(),
                                          "ids":ids_common}

    return relative_poses
    
def visualise_cameras_and_triangulated_points(views, minimal_tree, poses, triang_points, 
                                              max_points=100, path=None): 
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    all_points = []
    for i, (view1, view2) in enumerate(reversed(minimal_tree)):

        R1 = np.asarray(poses[view1]['R'], np.float64)
        t1 = np.asarray(poses[view1]['t'], np.float64).reshape(3,1)
        R2 = np.asarray(poses[view2]['R'], np.float64)
        t2 = np.asarray(poses[view2]['t'], np.float64).reshape(3,1)    

        points_3d = np.asarray(triang_points[(view1, view2)]['triang_points'], np.float64).T
        
        # pick some point at random
        idxs = np.arange(points_3d.shape[1])
        np.random.shuffle(idxs)
        idxs = idxs[:max_points] 
        points_3d = points_3d[:,idxs]

        _, t1_inv = utils.invert_Rt(R1, t1)
        _, t2_inv = utils.invert_Rt(R2, t2)  

        all_points.append(t1_inv.reshape(1,3))
        all_points.append(t2_inv.reshape(1,3))
        all_points.append(points_3d.T)   

        color_view1 = view_colors[views.index(view1)]
        #color_view2 = view_colors[views.index(view2)]

        ax.scatter(points_3d[0], points_3d[1], points_3d[2], c=np.array([color_view1]))
        
        p1 = t1_inv.ravel()
        p2 = t2_inv.ravel()
        ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]], 'k',alpha=1, linewidth=1)
        ax.scatter(*p1, c=np.array([color_view1]), marker='s', s=120, label=view1)
        ax.scatter(*p2, c=np.array([color_view1]), marker='x', s=250)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    all_points = np.vstack(all_points)
    x_min, y_min, z_min = np.min(all_points, axis=0)
    x_max, y_max, z_max = np.max(all_points, axis=0)

    ax.set_xlim(x_min-0.1*np.abs(x_min), x_max+0.1*np.abs(x_max))
    ax.set_ylim(y_min-0.1*np.abs(y_min), y_max+0.1*np.abs(y_max))
    ax.set_zlim(z_min-0.1*np.abs(z_min), z_max+0.1*np.abs(z_max))

    plt.legend()
    plt.show()
    
    if path is not None:
        utils.mkdir(path)
        ax.view_init(15, 0)
        plt.savefig(path+"/cameras_points3d_1.jpg", bbox_inches='tight')
        ax.view_init(15, 90)
        plt.savefig(path+"/cameras_points3d_2.jpg", bbox_inches='tight')
        ax.view_init(15+90, 0)
        plt.savefig(path+"/cameras_points3d_3.jpg", bbox_inches='tight')
        ax.view_init(15+90, 90)
        plt.savefig(path+"/cameras_points3d_4.jpg", bbox_inches='tight')
        ax.view_init(20, -125)
        plt.savefig(path+"/cameras_points3d_5.jpg", bbox_inches='tight')
        
def concatenate_relative_poses(minimal_tree, relative_poses, method='cross-ratios', verbose=2, print_prefix=''):
    
    # initialize the graph with the first pair of view
    # The first camera will be the center of our coordinate system for now
    pair0 = tuple(minimal_tree[0])
    poses = {pair0[0]: {"R":np.eye(3).tolist(), "t":np.zeros((3,1)).tolist()},
             pair0[1]: {"R":relative_poses[pair0]['Rd'],
                        "t":relative_poses[pair0]['td']}}
    triang_points = {pair0: {"triang_points":relative_poses[pair0]['triang_points'],
                             "ids":relative_poses[pair0]['ids']}}

    def find_adjacent_pair(pair):
        adj_pair = None
        inverse = False
        for key,data in triang_points.items():
            if pair[0] in key :
                adj_pair = key
            elif pair[1] in key:
                adj_pair = key
                inverse = True
                break 
        return adj_pair, inverse

    pairs = minimal_tree[1:]  

    while len(pairs)>0: 

        unmatched_pairs = []
        for curr_pair in pairs:
            curr_pair = tuple(curr_pair)

            adj_pair, inverse = find_adjacent_pair(curr_pair)

            if adj_pair is None:
                if curr_pair not in unmatched_pairs:
                    unmatched_pairs.append(curr_pair)
                continue
                
            if inverse:
                first_view, second_view = curr_pair[1], curr_pair[0]
            else:
                first_view, second_view = curr_pair

            # this is the new 0,0,0 point for the current pair
            R1 = np.asarray(poses[first_view]['R'], np.float32)
            t1 = np.asarray(poses[first_view]['t'], np.float32).reshape(3,1)

            # relative pose of the current pair
            Rd = np.asarray(relative_poses[curr_pair]['Rd'], np.float64)
            td = np.asarray(relative_poses[curr_pair]['td'], np.float64).reshape(3,1)
            if inverse:
                Rd, td = utils.invert_Rt(Rd, td)

            # triangulated points of the adjacent pair
            p3d_adj = np.float64(triang_points[adj_pair]['triang_points'])
            ids_adj = triang_points[adj_pair]['ids']

            # triangulated points of the current pair
            p3d = np.float64(relative_poses[curr_pair]['triang_points'])
            ids = relative_poses[curr_pair]['ids']
            
            # find common points
            p3d_adj_com, p3d_com, _ = _common_landmarks(p3d_adj, p3d, ids_adj, ids)                      

            # estimate scale between the two pairs
            if method=='cross-ratios':
                relative_scale, relative_scale_std = estimate_scale_point_sets(p3d_com, p3d_adj_com)
            elif method=='procrustes':
                relative_scale,_,_,_ = procrustes_registration(p3d_com, p3d_adj_com)
                _, relative_scale_std = estimate_scale_point_sets(p3d_com, p3d_adj_com)
            else:
                raise ValueError("Unrecognized method '{}'".format(method))

            # compute new camera pose for view2 of curr_pair
            R2 = np.dot(Rd, R1)
            t2 = np.dot(Rd, t1)+relative_scale*td   
            
            if verbose>0:
                logging.info("{}Concatenating relative poses for pair: {}".format(print_prefix, curr_pair))
                logging.info("{}\t Relative scale to {} n_points={}: {:0.3f}+-{:0.3f}".format(print_prefix, adj_pair, len(p3d_com), relative_scale, relative_scale_std))
                logging.info("{}\t {} new position: {}".format(print_prefix, second_view, utils.invert_Rt(R2, t2)[1].ravel()))
                
            if relative_scale<0.1:
                logging.info("!!!! The relative scale for this pair is quite low. Something may have gone wrong !!!!")                

            # transform the triangulated points of the current pair to the origin
            if inverse:
                R_inv, t_inv = utils.invert_Rt(R2, t2)        
            else:
                R_inv, t_inv = utils.invert_Rt(R1, t1)
                
            p3d = np.dot(R_inv, np.float64(p3d).T*relative_scale)+np.reshape(t_inv, (3,1))

            poses[second_view] = {'R':R2.tolist(), 't':t2.tolist()}
            triang_points[curr_pair] = {'triang_points':p3d.T.tolist(),
                                        "ids":ids}

        if len(pairs)==len(unmatched_pairs):
            raise RuntimeError("The following pairs are not connected to the rest of the network: {}".format(unmatched_pairs))
            break

        pairs = unmatched_pairs[:]

    return poses, triang_points

def build_view_graph(views, landmarks):

    G = nx.Graph()
    G.add_nodes_from(views)
    for view1, view2 in itertools.combinations(views, 2):

        pts1, pts2, _ = _common_landmarks(landmarks[view1]["landmarks"],
                                          landmarks[view2]["landmarks"],
                                          landmarks[view1]["ids"],
                                          landmarks[view2]["ids"])
        n_points = len(pts1)

        # checking if the pair of view has the minimum number 
        # of points for computing the fundamental matrix
        if n_points>=8:
            G.add_edge(view1, view2, n_points=n_points)   
            
    return G

def sample_random_view_tree(views, view0, landmarks):   

    G = build_view_graph(views, landmarks)

    for s,t,data in G.edges(data=True):
        data['weight'] = random.random()

    T = nx.maximum_spanning_tree(G)
    minimum_tree = list(nx.dfs_edges(T, source=view0))
    
    return minimum_tree

def compute_relative_poses_robust(views, view_tree, intrinsics, landmarks, 
                                  method='8point', th=1, max_paths=5, 
                                  method_scale='cross-ratios', verbose=2):
    
    G = build_view_graph(views, landmarks)
    
    relative_poses = {}
    for view1, view2 in view_tree:
        
        if verbose>0:
            logging.info("-------------------------------------------------")
            logging.info("Computing robust relative pose for pair {}->{}".format(view1, view2))
            logging.info("Initial relative pose:")
        
        view_tree = [[view1, view2]]
        relative_pose = compute_relative_poses(view_tree, intrinsics, landmarks, method, th, 
                                               verbose, print_prefix='')
        
        triangles = [ x for x in list(nx.all_simple_paths(G, view1, view2, 2)) if len(x)==3][:max_paths]
        if len(triangles)==0:
            logging.info("There are no other way to reach view {} from {}.".format(view2, view1))
        
        if verbose>0:
            logging.info("Number of additional paths found: {}".format(len(triangles)))
        
        relative_pose_robust = {'Rd':[relative_pose[(view1, view2)]['Rd']], 
                                'td':[relative_pose[(view1, view2)]['td']]}
        for nodes in triangles:
            
            view_tree = [nodes[:2],nodes[1:]]
            _relative_pose = compute_relative_poses(view_tree, intrinsics, landmarks, method, th, 
                                                   verbose, print_prefix='\t')
            pose, _ = concatenate_relative_poses(view_tree, _relative_pose, method_scale, 
                                                 verbose, print_prefix='\t')
        
            relative_pose_robust['Rd'].append(pose[view2]['R'])
            relative_pose_robust['td'].append(pose[view2]['t'])
                  
        Rd = np.mean(relative_pose_robust['Rd'], 0)
        td = np.mean(relative_pose_robust['td'], 0)
        
        pts1, pts2, ids_common = _common_landmarks(landmarks[view1]["landmarks"],
                                                   landmarks[view2]["landmarks"],
                                                   landmarks[view1]["ids"],
                                                   landmarks[view2]["ids"])
        
        K1 = np.float64(intrinsics[view1]['K'])
        K2 = np.float64(intrinsics[view2]['K'])
        dist1 = np.float64(intrinsics[view1]['dist'])
        dist2 = np.float64(intrinsics[view2]['dist'])
        
        pts1_undist = undistort_points(pts1, K1, dist1)
        pts2_undist = undistort_points(pts2, K2, dist2)        
        
        F = fundamental_from_relative_pose(Rd, td, K1, K2)
      
        tri = triangulate(pts1_undist, pts2_undist, 
                          K1=K1, R1=np.eye(3), t1=np.zeros(3), dist1=None,
                          K2=K2, R2=Rd, t2=td, dist2=None)

        relative_poses[(view1, view2)] = {'F':F.tolist(),
                                          'Rd':Rd.tolist(),
                                          'td':td.tolist(),
                                          'triang_points':tri.tolist(),
                                          'ids':ids_common}  
        if verbose>0:
            logging.info("Final relative pose:")
        _print_relative_pose_info(F, Rd, td, pts1_undist, pts2_undist, verbose, '')

    return relative_poses

def global_registration(ba_poses, ba_points, landmarks_global):
    
    src, dst, ids_common = _common_landmarks(ba_points['points_3d'],
                                             landmarks_global['landmarks_global'],
                                             ba_points['ids'],
                                             landmarks_global['ids'])
    
    logging.info("src points: bundle adjustment optimized 3d points")
    logging.info("dst points: global coordinates")    
    
    scale, R, t, mean_dist = point_set_registration(src, dst, verbose=True)
    
    global_triang_points = {'points_3d':apply_rigid_transform(src, R, t, scale).tolist(),
                            'ids': ids_common}
    
    R_inv, t_inv = utils.invert_Rt(R, t)

    global_poses = {}
    for cam,data in ba_poses.items():

        R = np.array(data['R'])
        t = np.reshape(data['t'], (3,1))

        R2 = np.dot(R, R_inv)
        t2 = np.dot(R, t_inv.reshape(3,-1)) + t.reshape(3,-1)*scale   

        global_poses[cam] = {'K':data['K'], 'dist':data['dist'],
                             'R':R2.tolist(), 't':t2.ravel().tolist()}
        
    return global_poses, global_triang_points

def visualise_global_registration(global_poses, landmarks_global, ba_poses, ba_points, 
                                  filenames, output_path="output/global_registration"):
    import matplotlib.pyplot as plt
    
    utils.mkdir(output_path)
    
    def plot(points3d, calib):
        rvec = cv2.Rodrigues(np.array(calib['R']))[0]
        tvec = np.array(calib['t'])
        K = np.array(calib['K'])
        dist = np.array(calib['dist'])

        return cv2.projectPoints(points3d, rvec, tvec, K, dist)[0].reshape(-1,2)    
    
    for view, calib in global_poses.items():
        
        img = imageio.imread(filenames[view].format(view))

        proj_glob = plot(np.array(landmarks_global['landmarks_global']), calib)
        proj_ba = plot(np.array(ba_points['points_3d']), ba_poses[view])

        plt.figure(figsize=(14,8))
        plt.plot(proj_ba[:,0], proj_ba[:,1], 'r.', label='B.A points')
        plt.plot(proj_glob[:,0], proj_glob[:,1], 'b.', label='Global points') 
        plt.imshow(img)
        plt.show()
        plt.legend()
        plt.savefig(os.path.join(output_path,"global_registration_{}.jpg".format(view)), bbox_inches='tight')