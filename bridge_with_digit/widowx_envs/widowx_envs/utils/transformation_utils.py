import numpy as np
import math

def TransToRp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector

    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        (np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]]),
         np.array([0, 0, 3]))
    """
    T = np.array(T)
    return T[0: 3, 0: 3], T[0: 3, 3]

def RpToTrans(R, p):
    """Converts a rotation matrix and a position vector into homogeneous
    transformation matrix

    :param R: A 3x3 rotation matrix
    :param p: A 3-vector
    :return: A homogeneous transformation matrix corresponding to the inputs

    Example Input:
        R = np.array([[1, 0,  0],
                      [0, 0, -1],
                      [0, 1,  0]])
        p = np.array([1, 2, 5])
    Output:
        np.array([[1, 0,  0, 1],
                  [0, 0, -1, 2],
                  [0, 1,  0, 5],
                  [0, 0,  0, 1]])
    """
    return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]

def TransInv(T):
    """Inverts a homogeneous transformation matrix

    :param T: A homogeneous transformation matrix
    :return: The inverse of T
    Uses the structure of transformation matrices to avoid taking a matrix
    inverse, for efficiency.

    Example input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1,  0, 0,  0],
                  [0,  0, 1, -3],
                  [0, -1, 0,  0],
                  [0,  0, 0,  1]])
    """
    R, p = TransToRp(T)
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]



# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R, thresh=1e-6):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < thresh


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R, check_error_thresh=1e-6):

    assert(isRotationMatrix(R, check_error_thresh))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def transform2state(eep, desired_gripper_position, default_rotation):
    assert eep.shape == (4, 4)
    if isinstance(desired_gripper_position, float):
        desired_gripper_position = np.array([desired_gripper_position])
    assert desired_gripper_position.shape == (1,)

    rot, xyz = TransToRp(eep)
    euler = rotationMatrixToEulerAngles(rot.dot(default_rotation.transpose()), check_error_thresh=1e-5)
    return np.concatenate([xyz, euler, desired_gripper_position])

def state2transform(state, default_rotation):
    assert state.shape == (7,)
    xyz, euler, gripper_state = state[:3], state[3:6], state[6:]
    trans = RpToTrans(eulerAnglesToRotationMatrix(euler).dot(default_rotation), xyz)
    return trans, gripper_state

def action2transform_local(actions, current_eef_position):
    """
    actions: [xyz deltas, and euler rotation angles, grasp_action], the rotation is around the position of the end-effector (axes are the same as world)
    current_eef_position: xyz of current end-effector

    return: transformation (4x4) and desired gripper state
    """
    assert current_eef_position.shape == (3,)
    assert actions.shape == (7,)

    Teef = RpToTrans(np.eye(3), current_eef_position)

    xyz, euler, gripper_state = actions[:3], actions[3:6], actions[6]
    trans = RpToTrans(eulerAnglesToRotationMatrix(euler), xyz)

    trans = Teef.dot(trans).dot(TransInv(Teef))
    return trans, gripper_state

def transform2action_local(transform, gripper_action, current_eef_position):
    """
    transform: 4x4 matrix, transform between current eef_pose in world coordinates to next eef_pose
    current_eef_position: xyz of current end-effector

    return: [xyz deltas, and euler rotation angles, grasp_action], the rotation is around the position of the end-effector (axes are the same as world)
    """
    assert current_eef_position.shape == (3,)
    assert transform.shape == (4, 4)
    Teef = RpToTrans(np.eye(3), current_eef_position)
    transform = TransInv(Teef).dot(transform).dot(Teef)

    if isinstance(gripper_action, float):
        gripper_action = np.array([gripper_action])
    assert gripper_action.shape == (1,)

    rot, xyz = TransToRp(transform)
    euler = rotationMatrixToEulerAngles(rot, check_error_thresh=1e-5)
    return np.concatenate([xyz, euler, gripper_action])
