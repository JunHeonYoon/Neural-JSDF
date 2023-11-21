#include "controller.h"

namespace NJSDF
{
    

    Controller::Controller()
    {
        if(Controller::init()) std::cout<<"Controller started"<<std::endl;
    }

    Controller::~Controller()
    {
        Controller::stopping();
    }

    bool Controller::init()
    {
        mode_change_thread_ = std::thread(&Controller::modeChangeReaderProc, this);

        q_.setZero();
        qdot_.setZero();
        q_init_.setZero();
        qdot_init_.setZero();
        q_desired_.setZero();
        qdot_desired_.setZero();
        tick_ = 0;

        robot_ = std::make_shared<RobotModel>();
        qp_ = std::make_shared<QP>(obs_radius_, njsdf_hz_);
        trigger_rate_ = std::make_shared<Timer>(njsdf_hz_);

        async_njsdf_thread_ = std::thread(&Controller::asyncNJSDFProc, this);


        return true;
    }

    void Controller::starting()
    {
        q_ = robot_->getJointPosition();
        play_time_ = 0;
    }

    void Controller::update()
    {
        if(is_first)
        {
            is_first = false;
            Controller::starting();
        }
        // std::cout<<"timer2: "<<timer_2.elapsedAndReset()*1000<<std::endl;
        NJSDF_input_mutex_.lock();
        q_ = robot_->getJointPosition();
        ee_pose_ = robot_->getTransformation(num_links);
        j_ = robot_->getJacobian(num_links);
        NJSDF_input_mutex_.unlock();

        if(calculation_mutex_.try_lock())
        {
            calculation_mutex_.unlock();
            if(async_calculation_thread_.joinable()) async_calculation_thread_.join();
            async_calculation_thread_ = std::thread(&Controller::asyncCalculationProc, this);
        }
        for(size_t i=0; i<9; ++i)
        {
            auto sleep_duration = std::chrono::duration<double>(1.0 / 30000);
            std::this_thread::sleep_for(sleep_duration);
            if(calculation_mutex_.try_lock())
            {
                calculation_mutex_.unlock();
                if(async_calculation_thread_.joinable()) async_calculation_thread_.join();
                break;
            }
        }

        // just put joint 
        q_ = q_desired_;
        qdot_ = qdot_desired_;
        robot_->getUpdateKinematics(q_, qdot_);

        if (DBG_CNT++ > hz_ / 5.)
        {
            DBG_CNT = 0;
            std::cout << "\n\n------------------------------------------------------------------" << std::endl;
            std::cout << "time     : " << std::fixed << std::setprecision(3) << play_time_ << std::endl;
            std::cout << "q now    :\t";
            std::cout << std::fixed << std::setprecision(3) << q_.transpose() << std::endl;
            std::cout << "q desired:\t";
            std::cout << std::fixed << std::setprecision(3) << q_desired_.transpose() << std::endl;
            std::cout << "x        :\t";
            std::cout << ee_pose_.translation().transpose() << std::endl;
            std::cout << "R        :\t" << std::endl;
            std::cout << std::fixed << std::setprecision(3) << ee_pose_.rotation() << std::endl;
            std::cout << "J        :\t" << std::endl;
            std::cout << std::fixed << std::setprecision(3) << j_ << std::endl;
            std::cout << "------------------------------------------------------------------\n\n" << std::endl;
        }

        tick_++;
        play_time_ = tick_ / hz_;
        // timer_2.reset();
    }

    void Controller::stopping()
    {
        std::cout << "NJSDF::Controller::stopping" << std::endl;
    }

    void Controller::modeChangeReaderProc()
    {
        while (!exit_flag_)
        {
            if(kbhit())
            {
                calculation_mutex_.lock();
                int key = getchar();
                switch (key)
                {
                    case 'h': // for home position joint ctrl
                        ctrl_mode_ = HOME;
                        mode_change_ = true;
                        break;
                    case 'n': // for NJSDF
                        ctrl_mode_ = NJSDF;
                        mode_change_ = true;
                        break;
                    case '\t':
                        if (is_simulation_run_) {
                            std::cout << "Simulation Pause" << std::endl;
                            is_simulation_run_ = false;
                        }
                        else {
                            std::cout << "Simulation Run" << std::endl;
                            is_simulation_run_ = true;
                        }
                        break;
                    case 'q':
                        is_simulation_run_ = false;
                        exit_flag_ = true;
                        Controller::~Controller();
                        break;
                    default:
                        ctrl_mode_ = NONE;
                        mode_change_ = true;
                        break;
                }
                calculation_mutex_.unlock();
            }
        }
    }

    void Controller::asyncCalculationProc()
    {
        calculation_mutex_.lock();
        
        if(ctrl_mode_ == HOME)
        {
            if(mode_change_)
            {
                std::cout << "================ Mode change: HOME position ================" <<std::endl;
                mode_change_ = false;
                njsdf_thread_enabled_ = false;
                q_init_ = q_;
                qdot_init_ = qdot_;
                control_start_time_ = play_time_;
                target_q_ << 0.0, 0.0, 0.0, -M_PI / 2., 0.0, M_PI / 2, M_PI / 4;
            }
            // -- moveJointPosition --
            Eigen::Matrix<double, NJSDF::dof, 1> zero_vector;
            zero_vector.setZero();
            q_desired_ = DyrosMath::cubicVector<7>(play_time_,
                                                    control_start_time_,
                                                    control_start_time_ + 2.0, q_init_, target_q_, zero_vector, zero_vector);
            qdot_desired_ = DyrosMath::cubicDotVector<7>(play_time_,
                                                            control_start_time_,
                                                            control_start_time_ + 2.0, q_init_, target_q_, zero_vector, zero_vector);
        }
        else if(ctrl_mode_ == NJSDF)
        {
            if(mode_change_)
            {
                std::cout << "================ Mode change: NJSDF ================" <<std::endl;
                mode_change_ = false;

                obs_posi << 0.5, 0.0, 0.45;
                // obs_posi << 0.5, 0.0, 100;
                target_ee_pose_ = ee_pose_;
                target_ee_pose_.translation() << 0.5, 0.5, 0.5;

                njsdf_thread_enabled_ = true;
                q_init_ = q_;
                qdot_init_ = qdot_;
                control_start_time_ = play_time_;
            }
            Eigen::Vector3d target_ee_vel;
            target_ee_vel << 0, -0.05, 0;

            target_ee_pose_.translation() += target_ee_vel / hz_;
            target_ee_pose_.translation()(1) = std::min(0.5, std::max(-0.5, target_ee_pose_.translation()(1)));
            // std::cout<<"target: "<<std::endl<<target_ee_pose_.rotation()<<"\n\n\n"<<std::endl;

            
            if(trigger_rate_->trigger())
            {
                // std::cout<<"timer1: " << timer_1.elapsedAndReset()*1000 <<std::endl;
                q_desired_ += opt_dq_ * (njsdf_hz_ / hz_);
                // timer_1.reset();
            }

        }
        else
        {
            q_desired_ = q_;
        }

        calculation_mutex_.unlock();
    }

    void Controller::asyncNJSDFProc()
    {
        while (!exit_flag_)
        {
            while(njsdf_thread_enabled_)
            {
                NJSDF_input_mutex_.lock();
                qp_->setCurrentState(ee_pose_, q_, j_);
                qp_->setObsPosition(obs_posi);
                qp_->setDesiredState(target_ee_pose_);
                NJSDF_input_mutex_.unlock();

                if(qp_->solveQP())
                {
                    NJSDF_mutex_.lock();
                    opt_dq_ = qp_->getJointDisplacement();
                    NJSDF_mutex_.unlock();
                }
                else
                {
                    std::cout<<"=======================not solved======================"<<std::endl;
                    opt_dq_.setZero();
                }

            }
        }
    }

}