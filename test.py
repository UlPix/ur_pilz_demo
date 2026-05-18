import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from moveit_msgs.srv import GetMotionSequence
from moveit_msgs.msg import (
    MotionSequenceRequest,
    MotionSequenceItem,
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    RobotState,
)
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import Quaternion, Pose
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
import tf2_ros


class PilzSequenceTest(Node):
    def __init__(self):
        super().__init__("pilz_sequence_test")

        # ── Callback group ────────────────────────────────────────────────────
        # ReentrantCallbackGroup lets callbacks (service responses, action
        # feedback, TF lookups, …) execute concurrently and re-enter each
        # other, which is required when an action send_goal callback itself
        # awaits another async call.
        self.cb_group = ReentrantCallbackGroup()

        # ── Pilz sequence service client ──────────────────────────────────────
        self.client = self.create_client(
            GetMotionSequence,
            "/plan_sequence_path",
            callback_group=self.cb_group,
        )
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("waiting for /plan_sequence_path …")

        # ── FollowJointTrajectory action client ───────────────────────────────
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
            callback_group=self.cb_group,
        )
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn(
                "FollowJointTrajectory action server not available – "
                "trajectory execution will be skipped."
            )

        # ── Joint-state subscriber ────────────────────────────────────────────
        self.current_joint_state = None
        self.js_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.js_callback,
            10,
            callback_group=self.cb_group,
        )

        # ── TF listener ───────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Wait for first joint state ────────────────────────────────────────
        while self.current_joint_state is None:
            self.get_logger().info("waiting for joint states …")
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info(
            f"got joint state: {self.current_joint_state.name}"
        )

        # ── Wait for TF ───────────────────────────────────────────────────────
        while not self.tf_buffer.can_transform(
            "base_link", "tool0", rclpy.time.Time()
        ):
            self.get_logger().info("waiting for TF base_link → tool0 …")
            rclpy.spin_once(self, timeout_sec=0.1)

        t = self.tf_buffer.lookup_transform(
            "base_link", "tool0", rclpy.time.Time()
        )

        self.cx = t.transform.translation.x
        self.cy = t.transform.translation.y
        self.cz = t.transform.translation.z
        self.cqx = t.transform.rotation.x
        self.cqy = t.transform.rotation.y
        self.cqz = t.transform.rotation.z
        self.cqw = t.transform.rotation.w

        self.get_logger().info(
            f"current TCP  : ({self.cx:.3f}, {self.cy:.3f}, {self.cz:.3f})"
        )
        self.get_logger().info(
            f"current ori  : ({self.cqx:.4f}, {self.cqy:.4f}, "
            f"{self.cqz:.4f}, {self.cqw:.4f})"
        )

        self.send_request()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def js_callback(self, msg: JointState):
        self.current_joint_state = msg

    # ── Helpers ───────────────────────────────────────────────────────────────

    def reorder_joint_state(self, js: JointState) -> JointState:
        correct_order = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        js_map = dict(zip(js.name, js.position))
        ordered = JointState()
        ordered.header = js.header
        ordered.name = correct_order
        ordered.position = [js_map[n] for n in correct_order if n in js_map]
        if len(ordered.position) != len(correct_order):
            self.get_logger().error(
                f"Missing joints! Got: {list(js_map.keys())}"
            )
        return ordered

    def make_motion_plan(
        self, x: float, y: float, z: float, is_first: bool = False
    ) -> MotionPlanRequest:
        req = MotionPlanRequest()
        req.group_name = "ur_manipulator"
        req.planner_id = "PTP"
        req.pipeline_id = "pilz_industrial_motion_planner"
        req.max_velocity_scaling_factor = 0.2
        req.max_acceleration_scaling_factor = 0.2   
        req.num_planning_attempts = 1
        req.allowed_planning_time = 2.0

        if is_first:
            start_state = RobotState()
            start_state.joint_state = self.reorder_joint_state(
                self.current_joint_state
            )
            req.start_state = start_state

        # Position constraint
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = "base_link"
        pos_constraint.link_name = "tool0"
        pos_constraint.weight = 1.0

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.001]

        sphere_pose = Pose()
        sphere_pose.position.x = x
        sphere_pose.position.y = y
        sphere_pose.position.z = z
        sphere_pose.orientation.w = 1.0

        bv = BoundingVolume()
        bv.primitives = [primitive]
        bv.primitive_poses = [sphere_pose]
        pos_constraint.constraint_region = bv

        # Orientation constraint (keep current TCP orientation)
        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = "base_link"
        ori_constraint.link_name = "tool0"
        ori_constraint.orientation = Quaternion(
            x=self.cqx, y=self.cqy, z=self.cqz, w=self.cqw
        )
        ori_constraint.absolute_x_axis_tolerance = 0.1
        ori_constraint.absolute_y_axis_tolerance = 0.1
        ori_constraint.absolute_z_axis_tolerance = 0.1
        ori_constraint.weight = 1.0

        goal = Constraints()
        goal.position_constraints = [pos_constraint]
        goal.orientation_constraints = [ori_constraint]
        req.goal_constraints = [goal]

        return req
    def make_circle_items(
        self,
        cx: float,
        cy: float,
        cz: float,
        radius: float = 0.05,
        n_points: int = 16,
        blend: float = 0.005,
    ) -> list:
        """
        Generate n_points PTP waypoints arranged in a circle of `radius` metres
        in the XY plane around (cx, cy, cz).
        
        The first point gets is_first=False because it follows the lift move.
        The last point gets blend=0.0 (full stop).
        """
        items = []
        for i in range(n_points):
            angle = 2.0 * math.pi * i / n_points      # evenly spaced angles
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            z = cz                                      # stay at same height

            is_last  = (i == n_points - 1)
            b = 0.0 if is_last else blend               # last point must stop

            items.append(
                self.make_item(x, y, z, blend=b, is_first=False)
            )
        return items



    def make_item(
        self,
        x: float,
        y: float,
        z: float,
        blend: float,
        is_first: bool = False,
    ) -> MotionSequenceItem:
        item = MotionSequenceItem()
        item.req = self.make_motion_plan(x, y, z, is_first=is_first)
        item.blend_radius = blend
        return item

    # ── Planning ──────────────────────────────────────────────────────────────

    def send_request(self):
        cx, cy, cz = self.cx, self.cy, self.cz
        self.get_logger().info(
            f"planning from ({cx:.3f}, {cy:.3f}, {cz:.3f}) upward in Z"
        )

        msg = GetMotionSequence.Request()
        seq = MotionSequenceRequest()
        # ── 1. Lift 2 cm ─────────────────────────────────────────────────────
        lift_1 = self.make_item(cx, cy, cz + 0.02, blend=0.01, is_first=True)

        # ── 2. Lift to circle height (5 cm), blend into first circle point ───
        lift_2 = self.make_item(cx, cy, cz + 0.05, blend=0.01, is_first=False)
        circle = self.make_circle_items(
            cx, cy, cz + 0.05,
            radius=0.1,
            n_points=20,
            blend=0.01,
        )
        seq.items = [lift_1, lift_2] + circle

        msg.request = seq

        self.get_logger().info("sending motion sequence …")
        future = self.client.call_async(msg)
        # planning_callback will fire inside the MultiThreadedExecutor thread
        # pool, so it is free to issue another async call (the action goal)
        # without deadlocking.
        future.add_done_callback(self.planning_callback)

    def planning_callback(self, future):
        """Called when the Pilz planner returns a result."""
        try:
            res = future.result()
        except Exception as exc:
            self.get_logger().error(f"Planning service error: {exc}")
            return

        self.get_logger().info(f"Error code     : {res.response.error_code.val}")
        self.get_logger().info(f"Planning time  : {res.response.planning_time:.3f} s")
        self.get_logger().info(
            f"Num trajectories: {len(res.response.planned_trajectories)}"
        )

        trajs = res.response.planned_trajectories
        if not trajs:
            self.get_logger().error("no trajectory returned – aborting")
            return

        traj = trajs[0]
        self.get_logger().info(
            f"success! trajectory points: "
            f"{len(traj.joint_trajectory.points)}"
        )

        # ── Kick off execution (callback-in-callback) ─────────────────────
        # Because we are using a ReentrantCallbackGroup and a
        # MultiThreadedExecutor, this send_goal_async call is safe here.
        self.send_trajectory(traj.joint_trajectory)

    # ── Execution ─────────────────────────────────────────────────────────────

    def send_trajectory(self, joint_trajectory: JointTrajectory):
        """Send a JointTrajectory to the FollowJointTrajectory action server."""
        if not self._action_client.server_is_ready():
            self.get_logger().error(
                "Action server not ready cannot execute trajectory."
            )
            return

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = joint_trajectory

        self.get_logger().info("sending goal to FollowJointTrajectory …")
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.execution_feedback_callback,
        )
        # execution_response_callback fires when the server accepts/rejects
        # the goal – again inside the executor thread pool.
        send_goal_future.add_done_callback(self.execution_response_callback)

    def execution_response_callback(self, future):
        """Called when the action server accepts or rejects the goal."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected by action server!")
            return

        self.get_logger().info("Goal accepted – waiting for result …")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.execution_result_callback)

    def execution_feedback_callback(self, feedback_msg):
        """Periodic progress updates from the controller."""
        fb = feedback_msg.feedback
        # FollowJointTrajectory feedback contains desired/actual/error fields.
        # Log the time elapsed along the trajectory.
        secs = fb.header.stamp.sec + fb.header.stamp.nanosec * 1e-9
        self.get_logger().info(
            f"[execution] t = {secs:.2f} s | "
            f"desired pos[0] = {fb.desired.positions[0]:.4f} rad"
            if fb.desired.positions
            else f"[execution] feedback received at t = {secs:.2f} s"
        )

    def execution_result_callback(self, future):
        """Called when trajectory execution finishes."""
        result = future.result().result
        status = future.result().status

        # FollowJointTrajectory result codes:
        #   0 = SUCCESSFUL, negative values = various error conditions
        error_code = result.error_code
        error_str  = result.error_string

        if error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info(
                f"Trajectory executed successfully (status={status})."
            )
        else:
            self.get_logger().error(
                f"Trajectory execution failed  "
                f"code={error_code}, msg='{error_str}' (status={status})."
            )


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = PilzSequenceTest()

    # MultiThreadedExecutor spins the node on a thread pool so that callbacks
    # triggered *inside* other callbacks (e.g. planning_callback → send_goal)
    # are dispatched to a free thread instead of deadlocking.
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()