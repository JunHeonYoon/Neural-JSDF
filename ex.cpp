#include "ex.h"
#include <fstream>

void AvatarController::setNeuralNetworks()
{
    ///// Between Left Arm and Upperbody & Head Collision Detection Network /////
    Eigen::VectorXd n_hidden, q_to_input_mapping_vector;
    n_hidden.resize(6);
    q_to_input_mapping_vector.resize(13);
    n_hidden << 120, 100, 80, 60, 40, 20;
    q_to_input_mapping_vector << 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24;
    initializeScaMlp(larm_upperbody_sca_mlp_, 13, 2, n_hidden, q_to_input_mapping_vector);
    loadScaNetwork(larm_upperbody_sca_mlp_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/sca_mlp/larm_upperbody/");
    //////////////////////////////////////////////////////////////////////////////

    ///// Between Right Arm and Upperbody & Head Collision Detection Network /////
    n_hidden << 120, 100, 80, 60, 40, 20;
    q_to_input_mapping_vector << 12, 13, 14, 25, 26, 27, 28, 29, 30, 31, 32, 23, 24;
    initializeScaMlp(rarm_upperbody_sca_mlp_, 13, 2, n_hidden, q_to_input_mapping_vector);
    loadScaNetwork(rarm_upperbody_sca_mlp_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/sca_mlp/rarm_upperbody/");
    //////////////////////////////////////////////////////////////////////////////

    ///// Between Arms Collision Detection Network /////
    // q_to_input_mapping_vector.resize(16);
    // n_hidden << 120, 100, 80, 60, 40, 20;
    // q_to_input_mapping_vector << 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32;
    // initializeScaMlp(btw_arms_sca_mlp_, 16, 2, n_hidden, q_to_input_mapping_vector);
    // loadScaNetwork(btw_arms_sca_mlp_, "/home/dyros/catkin_ws/src/tocabi_avatar/sca_mlp/btw_arms/");
    //////////////////////////////////////////////////////////////////////////////

    // // PETER GRU
    if (simulation_mode_)
    {
        initializeLegGRU(left_leg_peter_gru_, 30, 12, 150);
        loadGruWeightsSpectralNorm(left_leg_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/weights/left_leg/230221data_30_12_150_SN/");
        loadGruMeanStd(left_leg_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/mean_std/left_leg/tocabi_swing_ext_torque_230221_data/");

        initializeLegGRU(right_leg_peter_gru_, 30, 12, 150);
        loadGruWeightsSpectralNorm(right_leg_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/weights/right_leg/230221data_30_12_150_SN/");
        loadGruMeanStd(right_leg_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/mean_std/right_leg/tocabi_swing_ext_torque_230221_data/");

        initializeLegGRU(left_arm_peter_gru_, 34, 16, 200);
        loadGruWeightsSpectralNorm(left_arm_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/weights/left_arm/tocabi_230306data_q_qdot_34_16_200_SN_IS_32/");
        loadGruMeanStd(left_arm_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/mean_std/left_arm/tocabi_230306data_q_qdot_34_16_200_SN_IS_32/");

        initializeLegGRU(right_arm_peter_gru_, 34, 16, 200);
        loadGruWeightsSpectralNorm(right_arm_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/weights/right_arm/tocabi_230306data_q_qdot_34_16_200_SN_IS_32/");
        loadGruMeanStd(right_arm_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/mean_std/right_arm/tocabi_230306data_q_qdot_34_16_200_SN_IS_32/");

        initializeLegGRU(waist_peter_gru_, 50, 6, 200);
        loadGruWeightsSpectralNorm(waist_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/weights/waist/tocabi_230306data_q_qdot_34_6_200_SN_IS_32/");
        loadGruMeanStd(waist_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/mean_std/waist/tocabi_230306data_q_qdot_34_6_200_SN_IS_32/");

        initializeLegGRU(pelvis_peter_gru_, 86, 12, 200);
        loadGruWeightsSpectralNorm(pelvis_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/weights/pelvis/tocabi_230306data_q_qdot_34_16_200_SN_IS_64/");
        loadGruMeanStd(pelvis_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/mean_std/pelvis/tocabi_230306data_q_qdot_34_16_200_SN_IS_64/");
        // n_hidden.resize(2);
        // n_hidden << 128, 64;
        // initializeScaMlp(pelvis_concat_mlp_, 700, 12, n_hidden, q_to_input_mapping_vector, true);
        // pelvis_concat_mlp_.gaussian_mode = true;
        // loadMlpWeightsSpectralNorm(pelvis_concat_mlp_, CATKIN_WORKSPACE_DIR+"/src/tocabi_avatar/neural_networks/gru_tocabi/weights/pelvis/3layer_128_64/");
        // loadMlpMeanStd(pelvis_concat_mlp_, CATKIN_WORKSPACE_DIR+"/src/tocabi_avatar/neural_networks/gru_tocabi/mean_std/pelvis/");
    }
    else
    {
        initializeLegGRU(left_leg_peter_gru_, 30, 12, 150);
        loadGruWeightsSpectralNorm(left_leg_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/weights/left_leg/230221data_30_12_150_SN_IS_16/");
        loadGruMeanStd(left_leg_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/mean_std/left_leg/230221data_30_12_150_SN_IS_16/");

        initializeLegGRU(right_leg_peter_gru_, 30, 12, 150);
        loadGruWeightsSpectralNorm(right_leg_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/weights/right_leg/230221data_30_12_150_SN_IS_16/");
        loadGruMeanStd(right_leg_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/mean_std/right_leg/230221data_30_12_150_SN_IS_16/");

        initializeLegGRU(left_arm_peter_gru_, 34, 16, 200);
        loadGruWeightsSpectralNorm(left_arm_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/weights/left_arm/tocabi_230512data_q_qdot_mob_lpf_34_16_200_SN_IS_16/");
        loadGruMeanStd(left_arm_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/mean_std/left_arm/tocabi_230512data_q_qdot_mob_lpf_34_16_200_SN_IS_16/");

        initializeLegGRU(right_arm_peter_gru_, 34, 16, 200);
        loadGruWeightsSpectralNorm(right_arm_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/weights/right_arm/tocabi_230512data_q_qdot_mob_lpf_34_16_200_SN_IS_16/");
        loadGruMeanStd(right_arm_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/mean_std/right_arm/tocabi_230512data_q_qdot_mob_lpf_34_16_200_SN_IS_16/");

        initializeLegGRU(waist_peter_gru_, 50, 6, 200);
        loadGruWeightsSpectralNorm(waist_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/weights/waist/tocabi_230512data_q_qdot_mob_lpf_50_6_200_SN_IS_16/");
        loadGruMeanStd(waist_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/mean_std/waist/tocabi_230512data_q_qdot_mob_lpf_50_6_200_SN_IS_16/");

        initializeLegGRU(pelvis_peter_gru_, 74, 12, 200);
        loadGruWeightsSpectralNorm(pelvis_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/weights/pelvis/tocabi_230512data_mob_net_74_12_200_100step_SN_IS_32/");
        loadGruMeanStd(pelvis_peter_gru_, CATKIN_WORKSPACE_DIR + "/src/tocabi_avatar/neural_networks/gru_tocabi/mean_std/pelvis/tocabi_230512data_mob_net_74_12_200_100step_SN_IS_32/");

        // n_hidden.resize(2);
        // n_hidden << 128, 64;
        // initializeScaMlp(pelvis_concat_mlp_, 700, 12, n_hidden, q_to_input_mapping_vector, true);
        // pelvis_concat_mlp_.gaussian_mode = true;
        // loadMlpWeightsSpectralNorm(pelvis_concat_mlp_, CATKIN_WORKSPACE_DIR+"/src/tocabi_avatar/neural_networks/gru_tocabi/weights/pelvis/3layer_128_64/");
        // loadMlpMeanStd(pelvis_concat_mlp_, CATKIN_WORKSPACE_DIR+"/src/tocabi_avatar/neural_networks/gru_tocabi/mean_std/pelvis/");
    }
}
//////////Self Collision Avoidance Network////////////////
void AvatarController::initializeScaMlp(MLP &mlp, int n_input, int n_output, Eigen::VectorXd n_hidden, Eigen::VectorXd q_to_input_mapping_vector, bool spectral_normalization)
{
    mlp.n_input = n_input;
    mlp.n_output = n_output;
    mlp.n_hidden = n_hidden;
    mlp.n_layer = n_hidden.rows() + 1; // hiden layers + output layer
    mlp.q_to_input_mapping_vector = q_to_input_mapping_vector;

    mlp.weight.resize(mlp.n_layer);
    mlp.bias.resize(mlp.n_layer);
    mlp.hidden.resize(mlp.n_layer - 1);
    mlp.hidden_derivative.resize(mlp.n_layer - 1);

    if (spectral_normalization == false)
    {
        mlp.w_path.resize(mlp.n_layer);
        mlp.weight_files.resize(mlp.n_layer);
    }
    else
    {
        mlp.w_path.resize(3 * mlp.n_layer);
        mlp.weight_files.resize(3 * mlp.n_layer);
    }

    mlp.b_path.resize(mlp.n_layer);
    mlp.bias_files.resize(mlp.n_layer);
    //parameters resize
    for (int i = 0; i < mlp.n_layer; i++)
    {

        if (i == 0)
        {
            mlp.weight[i].setZero(mlp.n_hidden(i), mlp.n_input);
            mlp.bias[i].setZero(mlp.n_hidden(i));
            mlp.hidden[i].setZero(mlp.n_hidden(i));
            mlp.hidden_derivative[i].setZero(mlp.n_hidden(i), mlp.n_input);
        }
        else if (i == mlp.n_layer - 1)
        {
            mlp.weight[i].setZero(mlp.n_output, mlp.n_hidden(i - 1));
            mlp.bias[i].setZero(mlp.n_output);
        }
        else
        {
            mlp.weight[i].setZero(mlp.n_hidden(i), mlp.n_hidden(i - 1));
            mlp.bias[i].setZero(mlp.n_hidden(i));
            mlp.hidden[i].setZero(mlp.n_hidden(i));
            mlp.hidden_derivative[i].setZero(mlp.n_hidden(i), mlp.n_hidden(i - 1));
        }
    }

    //input output resize
    mlp.input_slow.setZero(mlp.n_input);
    mlp.input_fast.setZero(mlp.n_input);
    mlp.input_thread.setZero(mlp.n_input);

    mlp.output_slow.setZero(mlp.n_output);
    mlp.output_fast.setZero(mlp.n_output);
    mlp.output_thread.setZero(mlp.n_output);

    mlp.output_derivative_fast.setZero(mlp.n_output, mlp.n_input);
    mlp.hx_gradient_fast.setZero(mlp.n_input);
    mlp.hx_gradient_fast_lpf.setZero(mlp.n_input);
    mlp.hx_gradient_fast_pre.setZero(mlp.n_input);

    mlp.self_collision_stop_cnt_ = 0;
}
void AvatarController::loadScaNetwork(MLP &mlp, std::string folder_path)
{
    for (int i = 0; i < mlp.n_layer; i++)
    {
        mlp.w_path[i] = folder_path + "weight_" + std::to_string(i) + ".txt";
        mlp.b_path[i] = folder_path + "bias_" + std::to_string(i) + ".txt";

        mlp.weight_files[i].open(mlp.w_path[i], ios::in);
        mlp.bias_files[i].open(mlp.b_path[i], ios::in);

        readWeightFile(mlp, i);
        readBiasFile(mlp, i);
    }
}
void AvatarController::loadMlpWeightsSpectralNorm(MLP &mlp, std::string folder_path)
{
    for (int i = 0; i < mlp.n_layer; i++)
    {
        mlp.w_path[i] = folder_path + "weight_orig_" + std::to_string(i) + ".txt";
        mlp.w_path[i + mlp.n_layer] = folder_path + "weight_u_" + std::to_string(i) + ".txt";
        mlp.w_path[i + 2 * mlp.n_layer] = folder_path + "weight_v_" + std::to_string(i) + ".txt";

        mlp.b_path[i] = folder_path + "bias_" + std::to_string(i) + ".txt";

        mlp.weight_files[i].open(mlp.w_path[i], ios::in);
        mlp.weight_files[i + mlp.n_layer].open(mlp.w_path[i + mlp.n_layer], ios::in);
        mlp.weight_files[i + 2 * mlp.n_layer].open(mlp.w_path[i + 2 * mlp.n_layer], ios::in);

        mlp.bias_files[i].open(mlp.b_path[i], ios::in);

        readWeightFile(mlp, i, true);

        readBiasFile(mlp, i);
    }
}
void AvatarController::readWeightFile(MLP &mlp, int weight_num, bool spectral_normalization)
{
    if (!mlp.weight_files[weight_num].is_open())
    {
        std::cout << "Can not find the file: " << mlp.w_path[weight_num] << std::endl;
    }
    for (int i = 0; i < mlp.weight[weight_num].rows(); i++)
    {
        for (int j = 0; j < mlp.weight[weight_num].cols(); j++)
        {
            mlp.weight_files[weight_num] >> mlp.weight[weight_num](i, j);
        }
    }
    mlp.weight_files[weight_num].close();

    if (spectral_normalization)
    {
        Eigen::VectorXd W_i_u;
        Eigen::VectorXd W_i_v;
        W_i_u.setZero(mlp.weight[weight_num].rows());
        W_i_v.setZero(mlp.weight[weight_num].cols());

        for (int i = 0; i < mlp.weight[weight_num].rows(); i++)
        {
            mlp.weight_files[weight_num + mlp.n_layer] >> W_i_u(i);
        }

        for (int i = 0; i < mlp.weight[weight_num].cols(); i++)
        {
            mlp.weight_files[weight_num + 2 * mlp.n_layer] >> W_i_v(i);
        }

        double SN_W_i_temp = W_i_u.transpose() * mlp.weight[weight_num] * W_i_v;
        mlp.weight[weight_num] = mlp.weight[weight_num] / SN_W_i_temp;
    }

    if (mlp.loadweightfile_verbose == true)
    {
        cout << "weight_" << weight_num << ": \n"
             << mlp.weight[weight_num] << endl;
    }
}
void AvatarController::readBiasFile(MLP &mlp, int bias_num)
{
    if (!mlp.bias_files[bias_num].is_open())
    {
        std::cout << "Can not find the file: " << mlp.b_path[bias_num] << std::endl;
    }
    for (int i = 0; i < mlp.bias[bias_num].rows(); i++)
    {
        mlp.bias_files[bias_num] >> mlp.bias[bias_num](i);
    }
    mlp.bias_files[bias_num].close();

    if (mlp.loadbiasfile_verbose == true)
    {
        cout << "bias_" << bias_num - mlp.n_layer << ": \n"
             << mlp.bias[bias_num] << endl;
    }
}
void AvatarController::loadMlpMeanStd(MLP &mlp, std::string folder_path)
{
    std::string output_mean_path("output_mean.txt");
    std::string output_std_path("output_std.txt");

    output_mean_path = folder_path + output_mean_path;
    output_std_path = folder_path + output_std_path;

    mlp.mean_std_files[2].open(output_mean_path, ios::in);
    mlp.mean_std_files[3].open(output_std_path, ios::in);

    mlp.output_mean.setZero(mlp.n_output);
    mlp.output_std.setZero(mlp.n_output);

    // output_mean
    if (!mlp.mean_std_files[2].is_open())
    {
        std::cout << "Can not find the file: " << output_mean_path << std::endl;
    }

    for (int i = 0; i < mlp.n_output; i++)
    {
        mlp.mean_std_files[2] >> mlp.output_mean(i);
    }
    mlp.mean_std_files[2].close();

    // output_std
    if (!mlp.mean_std_files[3].is_open())
    {
        std::cout << "Can not find the file: " << output_std_path << std::endl;
    }

    for (int i = 0; i < mlp.n_output; i++)
    {
        mlp.mean_std_files[3] >> mlp.output_std(i);
    }
    mlp.mean_std_files[3].close();

    // gaussian mode
    if (mlp.gaussian_mode)
    {
        for (int i = int(mlp.n_output / 2); i < mlp.n_output; i++)
        {
            mlp.output_std(i) = 1.0;
            mlp.output_mean(i) = 0.0;
        }
    }

    // print
    if (false)
    {
        cout << "MLP output_mean: \n"
             << mlp.output_mean.transpose() << endl;

        cout << "MLP output_std: \n"
             << mlp.output_std.transpose() << endl;
    }
}
void AvatarController::calculateScaMlpInput(MLP &mlp)
{
    for (int i = 0; i < mlp.n_input; i++)
    {
        // mlp.input_slow(i) = rd_.q_(mlp.q_to_input_mapping_vector(i));
        // mlp.input_slow(i) = desired_q_fast_(mlp.q_to_input_mapping_vector(i));
        if (sca_dynamic_version_)
        {
            mlp.input_slow(i) = q_braking_stop_(mlp.q_to_input_mapping_vector(i));
        }
        else
        {
            mlp.input_slow(i) = rd_.q_(mlp.q_to_input_mapping_vector(i));
            // mlp.input_slow(i) = desired_q_fast_(mlp.q_to_input_mapping_vector(i));
        }
    }

    if (atb_mlp_input_update_ == false)
    {
        atb_mlp_input_update_ = true;
        mlp.input_thread = mlp.input_slow;
        q_ddot_max_thread_ = q_ddot_max_slow_;
        atb_mlp_input_update_ = false;
    }
}
void AvatarController::calculateScaMlpOutput(MLP &mlp)
{
    if (atb_mlp_input_update_ == false)
    {
        atb_mlp_input_update_ = true;
        mlp.input_fast = mlp.input_thread;
        q_ddot_max_fast_ = q_ddot_max_thread_;
        atb_mlp_input_update_ = false;
    }
    MatrixXd temp_derivative_pi;
    for (int layer = 0; layer < mlp.n_layer; layer++)
    {
        if (layer == 0) // input layer
        {
            mlp.hidden[0] = mlp.weight[0] * mlp.input_fast + mlp.bias[0];
            for (int h = 0; h < mlp.n_hidden(layer); h++)
            {
                mlp.hidden[0](h) = std::tanh(mlp.hidden[0](h));                                                       //activation function
                mlp.hidden_derivative[0].row(h) = (1 - (mlp.hidden[0](h) * mlp.hidden[0](h))) * mlp.weight[0].row(h); //derivative wrt input
            }
            temp_derivative_pi = mlp.hidden_derivative[0];
        }
        else if (layer == mlp.n_layer - 1) // output layer
        {
            mlp.output_fast = mlp.weight[layer] * mlp.hidden[layer - 1] + mlp.bias[layer];
            mlp.output_derivative_fast = mlp.weight[layer] * temp_derivative_pi;
        }
        else // hidden layers
        {
            mlp.hidden[layer] = mlp.weight[layer] * mlp.hidden[layer - 1] + mlp.bias[layer];
            for (int h = 0; h < mlp.n_hidden(layer); h++)
            {
                mlp.hidden[layer](h) = std::tanh(mlp.hidden[layer](h));                                                               //activation function
                mlp.hidden_derivative[layer].row(h) = (1 - (mlp.hidden[layer](h) * mlp.hidden[layer](h))) * mlp.weight[layer].row(h); //derivative wrt input
            }
            temp_derivative_pi = mlp.hidden_derivative[layer] * temp_derivative_pi;
        }
    }

    if (sca_dynamic_version_)
    {
        // for(int i=0; i<mlp.n_input; i++)
        // {
        // if(q_ddot_max_fast_(mlp.q_to_input_mapping_vector(i)) !=0 )
        // {
        //     mlp.output_derivative_fast.col(i) = mlp.output_derivative_fast.col(i)*
        //     ( 1 - desired_q_ddot_(mlp.q_to_input_mapping_vector(i))/q_ddot_max_fast_(mlp.q_to_input_mapping_vector(i)) );
        // }
        // }
    }

    mlp.hx_gradient_fast_pre = mlp.hx_gradient_fast;
    mlp.hx_gradient_fast = (mlp.output_derivative_fast.row(1) - mlp.output_derivative_fast.row(0)).transpose();
    for (int i = 0; i < mlp.n_input; i++)
    {
        mlp.hx_gradient_fast_lpf(i) = DyrosMath::lpf(mlp.hx_gradient_fast(i), mlp.hx_gradient_fast_pre(i), 1 / dt_, 10.0);
    }

    mlp.hx = mlp.output_fast(1) - mlp.output_fast(0);

    // if(atb_mlp_output_update_ == false)
    // {
    //     atb_mlp_output_update_ = true;
    //     mlp.output_thread = mlp.output_fast;
    //     atb_mlp_output_update_ = false;
    // }
}