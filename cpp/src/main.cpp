#include "NJSDF.h"
// #include "suhan_benchmark.h"
#include "robot.h"
#include "controller.h"

#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <thread>

// SuhanBenchmark bench_timer_;

// std::shared_ptr<NJSDF::NNmodel> njsdf_nnmodel_;
// std::shared_ptr<NJSDF::QP> njsdf_qp_;
// NJSDF::RobotModel *robot_;
std::shared_ptr<NJSDF::Controller> controller_;

// Eigen::Matrix<double, 3, 1> obs_position_;

// Eigen::Matrix<double, NJSDF::dof, 1> q_;
// Eigen::Matrix<double, NJSDF::dof, 1> qdot_;

// int kbhit(void)
// {
// 	struct termios oldt, newt;
// 	int ch;
// 	int oldf;

// 	tcgetattr(STDIN_FILENO, &oldt);
// 	newt = oldt;
// 	newt.c_lflag &= ~(ICANON | ECHO);
// 	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
// 	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
// 	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

// 	ch = getchar();

// 	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
// 	fcntl(STDIN_FILENO, F_SETFL, oldf);

// 	if(ch != EOF)
// 	{
// 	ungetc(ch, stdin);
// 	return 1;
// 	}

// 	return 0;
// }



int main()
{
    // -------------- Checking for Neural Network Model (DONE) ----------------------
    // njsdf_nnmodel_ = std::make_shared<NJSDF::NNmodel>();

    // Eigen::Matrix<double, 4, 1> n_hidden;
    // n_hidden << 256, 256, 256, 256;

    // njsdf_nnmodel_->setNeuralNetwork(dof+3, num_links, n_hidden, true);

    // q <<  0, 0, 0, -M_PI/2, 0, M_PI/2, M_PI/4;
    // obs_position << 0.555, 0, 0.55;

    // Eigen::Matrix<double, dof+3, 1> input;
    // input << q, obs_position;

    // bench_timer_.reset();
    // auto y_pred = njsdf_nnmodel_->calculateMlpOutput(input, true);
    // double elapsed_time = bench_timer_.elapsedAndReset();
    // std::cout << "minimum distance(cm): " << std::endl;
    // std::cout << y_pred.first.transpose() << std::endl;
    // std::cout << "minimum distance jac: " << std::endl;
    // std::cout << y_pred.second.block<num_links, dof>(0, 0) << std::endl;
    // std::cout << "elapsed_time: " << elapsed_time*1000.0 << std::endl;
    // ----------------------------------------------------------------------------

    // ------------------------- Checking for QP (DONE) ----------------------------------
    // double obs_radius = 0.05;
    // const int hz = 100;
    // njsdf_qp_ = std::make_shared<NJSDF::QP>(obs_radius, hz);
    // Eigen::Affine3d x_pose, desired_x_pose;
    // Eigen::Matrix<double, NJSDF::ee_dof, NJSDF::dof> j;

    // bench_timer_.reset();

    // // value for simulation
    // x_pose.translation() << 0.555, -0.017,  0.652; // sim: 0.652, real: 0.507
    // Eigen::AngleAxisd rotation(M_PI, Eigen::Vector3d::UnitX());
    // x_pose.rotate(rotation);
    // q_ << 0.000,  0.000,  0.000, -1.571,  0.000,  1.571,  0.785;
    // j << 0.017,  0.319,  0.017, -0.003, -0.000,  0.079, -0.017,
    //      0.555, -0.000,  0.555,  0.000,  0.079, -0.000,  0.000,
    //      0.000, -0.555, -0.000,  0.472, -0.017,  0.088,  0.000,
    //      0.000,  0.000,  0.000, -0.000,  1.000, -0.000, -0.000,
    //      0.000,  1.000, -0.000, -1.000, -0.000, -1.000,  0.000,
    //      1.000,  0.000,  1.000,  0.000, -0.001, -0.000, -1.000;
    
    // desired_x_pose = x_pose;
    // desired_x_pose.translation()[1] += 0.05 / hz;

    // obs_position_ << 0.555, 0, 0.55;

    // njsdf_qp_->setCurrentState(x_pose, q_, j);
    // njsdf_qp_->setDesiredState(desired_x_pose);
    // njsdf_qp_->setObsPosition(obs_position_);
    // if(njsdf_qp_->solveQP())
    // {
    //     Eigen::Matrix<double, NJSDF::dof, 1> opt_dq = njsdf_qp_->getJointDisplacement();
    //     std::cout<<"optimum delta q: "<<std::endl;
    //     std::cout<<opt_dq<<std::endl;
    // }
    // else
    // {
    //     std::cout<<"qp did not solved!!!! "<<std::endl;
    // }

    // double elapsed_time = bench_timer_.elapsedAndReset();
    // std::cout << "elapsed_time: " << elapsed_time*1000.0 << std::endl;
    // ----------------------------------------------------------------------------

    // ---------------------------- Checking for RBDL (DONE) -----------------------------

    // robot_ = new NJSDF::RobotModel();
    // const double hz = 1000.;

    // bool exit_flag = false;
    // bool is_first = true;
    // bool is_simulation_run = true;
    // bool is_mode_changed_ = false;
    // std::string control_mode_ = "default";

    // Eigen::Matrix<double, NJSDF::dof, 1> q_init_;
    // Eigen::Matrix<double, NJSDF::dof, 1> qdot_init_;
    // Eigen::Matrix<double, NJSDF::dof, 1> q_desired_;
    // Eigen::Matrix<double, NJSDF::dof, 1> qdot_desired_;
    // Eigen::Matrix<double, 3, 1> x_;
    // Eigen::Matrix<double, 3, 3> rotation_;

    // Eigen::Matrix<double, NJSDF::ee_dof, NJSDF::dof> j_;

    // double play_time_= 0.;
    // double control_start_time_ = 0.;
    // unsigned long tick_ = 0;


    

    // while(!exit_flag)
    // {
    //     bench_timer_.reset();
    //     if(is_first)
    //     {
    //         // init joint position 
    //         q_.setZero();
    //         is_first = false;
    //         std::cout<<"Initialized"<<std::endl;
    //     }
    //     if (kbhit())
	// 	{
	// 		int key = getchar();
	// 		switch (key)
	// 		{
    //             case 'h':
    //                 // -- ac.setMode("joint_ctrl_init") --
    //                 control_mode_ = "joint_ctrl_init";
    //                 is_mode_changed_ = true;
    //                 std::cout << "Current mode (changed) : " << control_mode_ << std::endl;
    //                 break;
    //             case '\t':
    //                 if (is_simulation_run) 
    //                 {
    //                     std::cout << "Simulation Pause" << std::endl;
    //                     is_simulation_run = false;
    //                 }
    //                 else 
    //                 {
    //                     std::cout << "Simulation Run" << std::endl;
    //                     is_simulation_run = true;
    //                 }
    //                 break;
    //             case 'q':
    //                 is_simulation_run = false;
    //                 exit_flag = true;
    //                 break;
    //             default:
    //                 break;
    //         }
    //     }

    //         if (is_simulation_run) 
    //         {
    //             // -- ac.compute() --
    //             robot_->getUpdateKinematics(q_, qdot_);
    //             x_ = robot_->getPosition(NJSDF::num_links);
    //             rotation_ = robot_->getOrientation(NJSDF::num_links);
    //             j_ = robot_->getJacobian(NJSDF::num_links);


    //             if(is_mode_changed_)
    //             {
    //                 is_mode_changed_ = false;

    //                 control_start_time_ = play_time_;

    //                 q_init_ = q_;
    //                 qdot_init_ = qdot_;
    //             }
    //             if(control_mode_ == "joint_ctrl_init")
    //             {
    //                 Eigen::Matrix<double, NJSDF::dof, 1> target_position;
	// 	            target_position << 0.0, 0.0, 0.0, -M_PI / 2., 0.0, M_PI / 2, M_PI / 4;
    //                 double duration = 1.0; 
    //                 // -- moveJointPosition --
    //                 Eigen::Matrix<double, NJSDF::dof, 1> zero_vector;
    //                 zero_vector.setZero();
    //                 q_desired_ = DyrosMath::cubicVector<7>(play_time_,
    //                                                        control_start_time_,
    //                                                        control_start_time_ + duration, q_init_, target_position, zero_vector, zero_vector);
    //                 qdot_desired_ = DyrosMath::cubicDotVector<7>(play_time_,
    //                                                              control_start_time_,
    //                                                              control_start_time_ + duration, q_init_, target_position, zero_vector, zero_vector);

    //             }

    //             // -- printState() --
    //             static int DBG_CNT = 0;
    //             if (DBG_CNT++ > hz / 100.)
    //             {
    //                 DBG_CNT = 0;
    //                 std::cout << "\n\n------------------------------------------------------------------" << std::endl;
    //                 std::cout << "time     : " << std::fixed << std::setprecision(3) << play_time_ << std::endl;
    //                 std::cout << "q now    :\t";
    //                 std::cout << std::fixed << std::setprecision(3) << q_.transpose() << std::endl;
    //                 std::cout << "q desired:\t";
    //                 std::cout << std::fixed << std::setprecision(3) << q_desired_.transpose() << std::endl;
    //                 std::cout << "x        :\t";
    //                 std::cout << x_.transpose() << std::endl;
    //                 std::cout << "R        :\t" << std::endl;
    //                 std::cout << std::fixed << std::setprecision(3) << rotation_ << std::endl;
    //                 std::cout << "J        :\t" << std::endl;
    //                 std::cout << std::fixed << std::setprecision(3) << j_ << std::endl;
                    
    //                 std::cout << "------------------------------------------------------------------\n\n" << std::endl;
	//             }
    //             tick_++;
    //             play_time_ = tick_ / hz;

    //             // -- vb.setDesiredPosition(ac.getDesiredPosition()) --
    //             q_ = q_desired_;
    //             qdot_ = qdot_desired_;

    //             auto sleep_duration = std::chrono::duration<double>(1.0 / hz);
    //             std::this_thread::sleep_for(sleep_duration);

    //             double elapsed_time = bench_timer_.elapsedAndReset();
    //             // std::cout << "elapsed_time: " << elapsed_time*1000.0 << std::endl;
    //         }
    // }
    // ----------------------------------------------------------------------------

    // ---------------------------- Checking for multi-threading -----------------------------
    controller_ = std::make_shared<NJSDF::Controller>();
    NJSDF::Timer trigger(controller_->hz_);
    
    while(true)
    {
        if(trigger.trigger() && controller_->is_simulation_run_) {
            controller_->update();
        }
        
    }

    // ---------------------------------------------------------------------------------------


    


    return 0;
}