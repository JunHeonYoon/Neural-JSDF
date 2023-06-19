// #include "math_type_define.h"

#include <std_msgs/String.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>

#include <sstream>
#include <fstream>


#include <iomanip>
#include <iostream>

// pedal
#include <std_msgs/Float32.h>

#include <eigen_conversions/eigen_msg.h>
using namespace std;

const string CATKIN_WORKSPACE_DIR= "/home/dg/catkin_ws";
const bool simulation_mode_ = true;
bool sca_dynamic_version_ = false;

Eigen::VectorQd q_braking_stop_;
RobotData &rd_;
//sca
Eigen::VectorQd q_ddot_max_slow_;
Eigen::VectorQd q_ddot_max_fast_;
Eigen::VectorQd q_ddot_max_thread_;
Eigen::VectorQd q_braking_stop_;
Eigen::VectorQd torque_max_braking_;

//////////////Self Collision Avoidance Network////////////////
class AvatarController
{
    /////////////////////////PETER GRU///////////////////////////////////////////////////////
    // GRU c++
    struct GRU
    {
        ~GRU() { std::cout << "GRU terminates" << std::endl; }
        std::atomic<bool> atb_gru_input_update_{false};
        std::atomic<bool> atb_gru_output_update_{false};

        int n_input;
        int n_output;
        int n_hidden;

        int buffer_size;
        Eigen::VectorXd ring_buffer; //ring_buffer
        int buffer_head;
        int buffer_tail;

        int input_mode_idx;
        int output_mode_idx;

        Eigen::MatrixXd input_slow;
        Eigen::MatrixXd input_fast;
        Eigen::MatrixXd input_thread;
        Eigen::VectorXd input_mean;
        Eigen::VectorXd input_std;

        Eigen::VectorXd robot_input_data;

        Eigen::VectorXd output;
        Eigen::VectorXd real_output; //real out with physical dimension
        Eigen::VectorXd output_mean;
        Eigen::VectorXd output_std;

        Eigen::MatrixXd W_ih;   // [ W_ir | W_iz | W_in ] R^{3*n_hidden X n_input}
        Eigen::VectorXd b_ih;   // [ b_ir | b_iz | b_in ]
        Eigen::MatrixXd W_hh;   // [ W_hr | W_hz | W_hn ]
        Eigen::VectorXd b_hh;   // [ b_hr | b_hz | b_hn ]
        Eigen::MatrixXd W_linear;
        Eigen::VectorXd b_linear;

        Eigen::MatrixXd h_t;    // hidden
        Eigen::VectorXd r_t;    // reset
        Eigen::VectorXd z_t;    // update
        Eigen::VectorXd n_t;    // new gates

        ifstream network_weights_files[3];
        ifstream bias_files[3];
        ifstream mean_std_files[4];

        bool loadweightfile_verbose = false;
        bool loadmeanstdfile_verbose = false;
        bool gaussian_mode = true;
    };
    GRU left_leg_peter_gru_;
    GRU right_leg_peter_gru_;
    GRU left_arm_peter_gru_;
    GRU right_arm_peter_gru_;
    GRU waist_peter_gru_;
    GRU pelvis_peter_gru_;

    const int gru_hz_ = 1000;
    // ifstream network_weights_file_gru_[6];
    // ifstream mean_std_file_gru_[4];

    Eigen::Vector6d estimated_ext_force_lfoot_gru_;
    Eigen::Vector6d estimated_ext_force_rfoot_gru_;
    Eigen::Vector6d estimated_ext_force_lhand_gru_;
    Eigen::Vector6d estimated_ext_force_rhand_gru_;

    Eigen::Vector6d estimated_ext_force_lfoot_gru_lpf_;
    Eigen::Vector6d estimated_ext_force_rfoot_gru_lpf_;

    Eigen::Vector6d estimated_ext_force_lfoot_gru_local_;
    Eigen::Vector6d estimated_ext_force_rfoot_gru_local_;
    Eigen::Vector6d estimated_ext_force_lfoot_gru_lpf_local_;
    Eigen::Vector6d estimated_ext_force_rfoot_gru_lpf_local_;

    Eigen::VectorVQd estimated_model_unct_torque_gru_fast_;
    Eigen::VectorVQd estimated_model_unct_torque_gru_slow_;
    Eigen::VectorVQd estimated_model_unct_torque_gru_thread_;
    Eigen::VectorVQd estimated_model_unct_torque_gru_slow_lpf_;

    Eigen::VectorVQd estimated_model_unct_torque_variance_gru_fast_;
    Eigen::VectorVQd estimated_model_unct_torque_variance_gru_slow_;
    Eigen::VectorVQd estimated_model_unct_torque_variance_gru_thread_;

    VectorVQd estimated_model_unct_torque_std_;
    VectorVQd estimated_model_unct_torque_std_lpf_soft_;

    Eigen::VectorVQd estimated_external_torque_gru_slow_;

    Eigen::VectorVQd estimated_external_torque_gru_slow_lpf_soft_;
    Eigen::VectorVQd estimated_external_torque_gru_slow_lpf_hard_;

    Vector6d pelv_pure_ext_torque_;
    Vector6d pelv_pure_ext_torque_lpf_soft_;
    Vector6d pelv_pure_ext_torque_lpf_hard_;

    bool check_left_swing_foot_;
    bool check_left_early_contact_;
    bool check_right_swing_foot_;
    bool check_right_early_contact_;

    int left_early_contact_cnt_;
    int right_early_contact_cnt_;
    
    bool left_leg_in_unexpected_collision_;
    bool right_leg_in_unexpected_collision_;
    bool left_arm_in_unexpected_collision_;
    bool right_arm_in_unexpected_collision_;
    bool upper_body_in_unexpected_collision_;
    bool pelv_in_unexpected_collision_;

    void collectRobotInputData_peter_gru();

    void loadGruWeights(GRU &gru, std::string folder_path);
    void loadGruMeanStd(GRU &gru, std::string folder_path);
    void loadGruWeightsSpectralNorm(GRU &gru, std::string folder_path);

    void initializeLegGRU(GRU &gru, int n_input, int n_output, int n_hidden);
    void calculateGruInput(GRU &gru, double input_scale);
    void calculateGruOutput(GRU &gru);

    Eigen::VectorXd vecSigmoid(VectorXd input);
    Eigen::VectorXd vecTanh(VectorXd input);
    //////////////////////////////////////////////////////////////////////////////////////////

    struct MLP
    {
        ~MLP() { std::cout << "MLP terminate" << std::endl; }
        std::vector<Eigen::MatrixXd> weight;
        std::vector<Eigen::VectorXd> bias;
        std::vector<Eigen::VectorXd> hidden;
        std::vector<Eigen::MatrixXd> hidden_derivative;

        std::vector<std::string> w_path;
        std::vector<std::string> b_path;

        std::vector<ifstream> weight_files;
        std::vector<ifstream> bias_files;
        ifstream mean_std_files[4];
        
        int n_input;
        int n_output;
        Eigen::VectorXd n_hidden;
        int n_layer;
        
        Eigen::VectorXd q_to_input_mapping_vector;

        Eigen::VectorXd input_slow;
        Eigen::VectorXd input_fast;
        Eigen::VectorXd input_thread;

        Eigen::VectorXd output_slow;
        Eigen::VectorXd output_fast;
        Eigen::VectorXd output_thread;

        Eigen::VectorXd output_mean;
        Eigen::VectorXd output_std;

        Eigen::MatrixXd output_derivative_fast;

        Eigen::VectorXd hx_gradient_fast;
        Eigen::VectorXd hx_gradient_fast_lpf;
        Eigen::VectorXd hx_gradient_fast_pre;

        double hx;

        bool loadweightfile_verbose = false;
        bool loadbiasfile_verbose = false;
        bool gaussian_mode = false;

        int self_collision_stop_cnt_;
    }   larm_upperbody_sca_mlp_, rarm_upperbody_sca_mlp_, btw_arms_sca_mlp_;
    
    MLP pelvis_concat_mlp_;

    void setNeuralNetworks();
    void initializeScaMlp(MLP &mlp, int n_input, int n_output, Eigen::VectorXd n_hidden, Eigen::VectorXd q_to_input_mapping_vector, bool spectral_normalization = false);
    void loadScaNetwork(MLP &mlp, std::string folder_path);
    void loadMlpWeightsSpectralNorm(MLP &mlp, std::string folder_path);

    void calculateScaMlpInput(MLP &mlp);
    void calculateScaMlpOutput(MLP &mlp);
    void calculatePelvConcatMlpOutput(MLP &mlp);

    void readWeightFile(MLP &mlp, int weight_num, bool spectral_normalization = false);
    void readBiasFile(MLP &mlp, int bias_num);
    void loadMlpMeanStd(MLP &mlp, std::string folder_path);

    std::atomic<bool> atb_mlp_input_update_{false};
    std::atomic<bool> atb_mlp_output_update_{false};

    Eigen::MatrixXd q_dot_buffer_slow_;  //20 stacks
    Eigen::MatrixXd q_dot_buffer_fast_;  //20 stacks
    Eigen::MatrixXd q_dot_buffer_thread_;  //20 stacks

    //////////////////////////////////////////////////////////////
}
