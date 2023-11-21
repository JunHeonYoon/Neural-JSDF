#ifndef NJSDF_CONTROLLER_
#define NJSDF_CONTROLLER_

#include <thread>
#include <mutex>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

#include "NJSDF.h"
#include "robot.h"
#include "common.h"
#include "suhan_benchmark.h"

namespace NJSDF
{
    class Timer 
    {
        public:
        explicit Timer(double rate = 30.0) : period_(1.0 / rate) {}
    
        bool trigger() 
        {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = current_time - time_stamp_;
    
        if (elapsed_time.count() >= period_) 
        {
            time_stamp_ = current_time;
            return true;
        }
    
        return false;
        }
    
        private:
        std::chrono::high_resolution_clock::time_point time_stamp_;
        double period_;
    };

    enum CTRL_MODE{NONE, HOME, NJSDF};
    
    class Controller
    {
        public:
            Controller();
            ~Controller();

            bool init();
            void starting();
            void update();
            void stopping();

            void modeChangeReaderProc();
            void asyncCalculationProc(); // hqp calculation
            void asyncNJSDFProc();

            int kbhit(void)
            {
                struct termios oldt, newt;
                int ch;
                int oldf;

                tcgetattr(STDIN_FILENO, &oldt);
                newt = oldt;
                newt.c_lflag &= ~(ICANON | ECHO);
                tcsetattr(STDIN_FILENO, TCSANOW, &newt);
                oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
                fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

                ch = getchar();

                tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
                fcntl(STDIN_FILENO, F_SETFL, oldf);

                if(ch != EOF)
                {
                ungetc(ch, stdin);
                return 1;
                }

                return 0;
            }

            bool mode_change_ = true;
            CTRL_MODE ctrl_mode_{NONE};
            bool exit_flag_ = false;
            bool is_simulation_run_ = true;
            double play_time_= 0.;
            double control_start_time_;
            unsigned long tick_;
            const double njsdf_hz_{100.0};
            const double hz_{1000.0};
            bool is_first = true;
            int DBG_CNT = 0;

            SuhanBenchmark timer_1, timer_2;


        // private:

            Eigen::Matrix<double, NJSDF::dof, 1> q_;
            Eigen::Matrix<double, NJSDF::dof, 1> qdot_;
            Eigen::Matrix<double, NJSDF::dof, 1> q_init_;
            Eigen::Matrix<double, NJSDF::dof, 1> qdot_init_;
            Eigen::Matrix<double, NJSDF::dof, 1> q_desired_;
            Eigen::Matrix<double, NJSDF::dof, 1> qdot_desired_;
            Eigen::Matrix<double, NJSDF::dof, 1> target_q_;
            Eigen::Matrix<double, NJSDF::dof, 1> opt_dq_;
            Eigen::Affine3d target_ee_pose_;
            Eigen::Affine3d ee_pose_;
            Eigen::Matrix<double, NJSDF::ee_dof, NJSDF::dof> j_;
            
            double obs_radius_{0.05};
            Eigen::Matrix<double, 3, 1> obs_posi;


            std::shared_ptr<RobotModel> robot_;
            std::shared_ptr<QP> qp_;
            std::shared_ptr<Timer> trigger_rate_ ;

            // Thread
            std::thread mode_change_thread_;
            std::thread async_calculation_thread_;
            std::thread async_njsdf_thread_;
            bool njsdf_thread_enabled_{false};

            // Mutex
            std::mutex calculation_mutex_;
            std::mutex NJSDF_mutex_;
            std::mutex NJSDF_input_mutex_;

            // bool quit_all_proc_{false};

            
    };
}

#endif // NJSDF_CONTROLLER_