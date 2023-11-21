#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <eigen3/Eigen/Eigen>
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <ctime>
#include <omp.h>

const std::string NJSDF_DIR = "/home/yoonjunheon/git/Neural-JSDF";

double ReLU(double input)
{
    if(input > 0)
    {
        return input;
    }
    else
    {
        return 0.;
    }
}

double ReLU_derivative(double input)
{
    if(input > 0)
    {
        return 1.;
    }
    else
    {
        return 0.;
    }
}

struct MLP
{
    ~MLP() { std::cout << "MLP terminate" << std::endl; }
    std::vector<Eigen::SparseMatrix<double>> weight;
    std::vector<Eigen::VectorXd> bias;
    std::vector<Eigen::VectorXd> hidden;
    std::vector<Eigen::SparseMatrix<double>> hidden_derivative;

    std::vector<std::string> w_path;
    std::vector<std::string> b_path;

    std::vector<std::ifstream> weight_files;
    std::vector<std::ifstream> bias_files;

    int n_input;
    int n_output;
    Eigen::VectorXd n_hidden;
    int n_layer;

    Eigen::VectorXd input;
    Eigen::VectorXd output;
    Eigen::MatrixXd output_derivative;

    bool is_nerf;
    Eigen::VectorXd input_nerf;
    Eigen::SparseMatrix<double> nerf_jac;

    bool loadweightfile_verbose = false;
    bool loadbiasfile_verbose = false;
}njsdf_;

void readWeightFile(MLP &mlp, int weight_num)
{
    if (!mlp.weight_files[weight_num].is_open())
    {
        std::cout << "Can not find the file: " << mlp.w_path[weight_num] << std::endl;
    }
    for (int i = 0; i < mlp.weight[weight_num].rows(); i++)
    {
        for (int j = 0; j < mlp.weight[weight_num].cols(); j++)
        {
            mlp.weight_files[weight_num] >> mlp.weight[weight_num].coeffRef(i, j);
        }
    }
    mlp.weight_files[weight_num].close();

    if (mlp.loadweightfile_verbose == true)
    {
        std::cout << "weight_" << weight_num << ": \n"
             << mlp.weight[weight_num] <<std::endl;
    }
}

void readBiasFile(MLP &mlp, int bias_num)
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
        std::cout << "bias_" << bias_num - mlp.n_layer << ": \n"
             << mlp.bias[bias_num] << std::endl;
    }
}

void loadNetwork(MLP &mlp, std::string folder_path)
{
    for (int i = 0; i < mlp.n_layer; i++)
    {
        mlp.w_path[i] = folder_path + "weight_" + std::to_string(i) + ".txt";
        mlp.b_path[i] = folder_path + "bias_" + std::to_string(i) + ".txt";

        mlp.weight_files[i].open(mlp.w_path[i], std::ios::in);
        mlp.bias_files[i].open(mlp.b_path[i], std::ios::in);

        readWeightFile(mlp, i);
        readBiasFile(mlp, i);
    }
}

void initializeNetwork(MLP &mlp, int n_input, int n_output, Eigen::VectorXd n_hidden, bool is_nerf)
{
    mlp.is_nerf = is_nerf;
    mlp.n_input = n_input;
    mlp.n_output = n_output;
    mlp.n_hidden = n_hidden;
    mlp.n_layer = n_hidden.rows() + 1; // hiden layers + output layer

    mlp.weight.resize(mlp.n_layer);
    mlp.bias.resize(mlp.n_layer);
    mlp.hidden.resize(mlp.n_layer - 1);
    mlp.hidden_derivative.resize(mlp.n_layer - 1);

    mlp.w_path.resize(mlp.n_layer);
    mlp.b_path.resize(mlp.n_layer); 
    mlp.weight_files.resize(mlp.n_layer);
    mlp.bias_files.resize(mlp.n_layer); 

    //parameters resize
    for (int i = 0; i < mlp.n_layer; i++)
    {
        if (i == 0)
        {
            if(mlp.is_nerf) 
            {
                mlp.weight[i].resize(mlp.n_hidden(i), 3 * mlp.n_input);
                mlp.weight[i].setZero();
                mlp.hidden_derivative[i].resize(mlp.n_hidden(i), 3 * mlp.n_input);
                mlp.hidden_derivative[i].setZero();
            }
            else
            {
                mlp.weight[i].resize(mlp.n_hidden(i), mlp.n_input);
                mlp.weight[i].setZero();
                mlp.hidden_derivative[i].resize(mlp.n_hidden(i), mlp.n_input);
                mlp.hidden_derivative[i].setZero();
            }
            mlp.bias[i].setZero(mlp.n_hidden(i));
            mlp.hidden[i].setZero(mlp.n_hidden(i));
        }
        else if (i == mlp.n_layer - 1)
        {
            mlp.weight[i].resize(mlp.n_output, mlp.n_hidden(i - 1));
            mlp.weight[i].setZero();
            mlp.bias[i].setZero(mlp.n_output);
        }
        else
        {
            mlp.weight[i].resize(mlp.n_hidden(i), mlp.n_hidden(i - 1));
            mlp.weight[i].setZero();
            mlp.bias[i].setZero(mlp.n_hidden(i));
            mlp.hidden[i].setZero(mlp.n_hidden(i));
            mlp.hidden_derivative[i].resize(mlp.n_hidden(i), mlp.n_hidden(i - 1));
            mlp.hidden_derivative[i].setZero();
        }
    }
    mlp.nerf_jac.resize(3 * njsdf_.n_input, njsdf_.n_input);
    mlp.nerf_jac.setZero();


    //input output resize
    mlp.input.resize(mlp.n_input);
    mlp.input_nerf.resize(3 * mlp.n_input);
    mlp.output.resize(mlp.n_output);
    mlp.output_derivative.setZero(mlp.n_output, mlp.n_input);
}

void setNeuralNetwork()
{
    Eigen::VectorXd n_hidden;
    n_hidden.resize(4);
    n_hidden << 256, 256, 256, 256;
    initializeNetwork(njsdf_, 10, 9, n_hidden, true);
    loadNetwork(njsdf_, NJSDF_DIR + "/learning/nn-learning/parameter/");
    Eigen::initParallel();
    Eigen::setNbThreads(8);
    std::cout<<"Thread: "<<Eigen::nbThreads()<<std::endl;
}

void setNetworkInput(Eigen::VectorXd input)
{
    njsdf_.input = input;
    if (njsdf_.is_nerf)
    {
        Eigen::VectorXd sinInput = input.array().sin();
        Eigen::VectorXd cosInput = input.array().cos();

        njsdf_.input_nerf.segment(0 * njsdf_.n_input, njsdf_.n_input) = input;
        njsdf_.input_nerf.segment(1 * njsdf_.n_input, njsdf_.n_input) = sinInput;
        njsdf_.input_nerf.segment(2 * njsdf_.n_input, njsdf_.n_input) = cosInput;
    }
    
    // std::cout<< "INPUT DATA:"<< std::endl <<njsdf_.input.transpose() << std::endl;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> calculateMlpOutput(bool time_verbose)
{
    // std::vector<clock_t> start, finish;
    // start.resize(3*njsdf_.n_layer);
    // finish.resize(3*njsdf_.n_layer);

    // start[3*njsdf_.n_layer - 1] = clock(); // Total 
    // Eigen::SparseMatrix<double> temp_derivative;
    // for (int layer = 0; layer < njsdf_.n_layer; layer++)
    // {
    //     if (layer == 0) // input layer
    //     {
    //         start[0] = clock(); // Linear 
    //         if (njsdf_.is_nerf) njsdf_.hidden[0] = njsdf_.weight[0] * njsdf_.input_nerf + njsdf_.bias[0];
    //         else                njsdf_.hidden[0] = njsdf_.weight[0] * njsdf_.input + njsdf_.bias[0];
    //         finish[0] = clock();
            
    //         start[1] = clock(); // ReLU
    //         njsdf_.hidden[0] = njsdf_.hidden[0].unaryExpr(&ReLU); //activation function
    //         njsdf_.hidden_derivative[layer].setZero();
    //         njsdf_.hidden_derivative[0] = njsdf_.hidden_derivative[0].transpose();
    //         for (int h = 0; h < njsdf_.n_hidden(0); h++)
    //         {                                                     
    //             if (njsdf_.hidden[0](h) > 0)
    //             {
    //                 njsdf_.hidden_derivative[0].col(h) =  njsdf_.weight[0].transpose().col(h); //derivative wrt input
    //             }
    //         }
    //         njsdf_.hidden_derivative[0] = njsdf_.hidden_derivative[0].transpose();
    //         finish[1] = clock();

    //         if (njsdf_.is_nerf)
    //         {
    //             for (int i = 0; i < njsdf_.n_input; i++)
    //             {
    //                 njsdf_.nerf_jac.coeffRef(0*njsdf_.n_input + i, i) = 1;
    //                 njsdf_.nerf_jac.coeffRef(1*njsdf_.n_input + i, i) = cos(njsdf_.input(i));
    //                 njsdf_.nerf_jac.coeffRef(2*njsdf_.n_input + i, i) = -sin(njsdf_.input(i));
    //             }
    //             start[2] = clock(); // Multip
    //             temp_derivative = (njsdf_.hidden_derivative[0] * njsdf_.nerf_jac);
    //             finish[2] = clock();
    //         }
    //         else
    //         {
    //             temp_derivative = njsdf_.hidden_derivative[0];
    //         }
    //     }
    //     else if (layer == njsdf_.n_layer - 1) // output layer
    //     {
    //         start[layer*3] = clock(); // Linear
    //         njsdf_.output = njsdf_.weight[layer] * njsdf_.hidden[layer - 1] + njsdf_.bias[layer];
    //         finish[layer*3] = clock(); 

    //         start[layer*3+1] = clock(); // Multip
    //         njsdf_.output_derivative = njsdf_.weight[layer] * temp_derivative;
    //         finish[layer*3+1] = clock();
    //     }
    //     else // hidden layers
    //     {
    //         start[layer*3] = clock(); // Linear
    //         njsdf_.hidden[layer] = njsdf_.weight[layer] * njsdf_.hidden[layer - 1] + njsdf_.bias[layer];
    //         finish[layer*3] = clock();

    //         start[layer*3+1] = clock(); // ReLU
    //         njsdf_.hidden[layer] = njsdf_.hidden[layer].unaryExpr(&ReLU); //activation function
    //         njsdf_.hidden_derivative[layer].setZero();
    //         // njsdf_.hidden_derivative[layer] = njsdf_.hidden_derivative[layer].transpose();
    //         for (int h = 0; h < njsdf_.n_hidden(layer); h++)
    //         {                                                     
    //             if (njsdf_.hidden[layer](h) > 0)
    //             {
    //                 // njsdf_.hidden_derivative[layer].col(h) =  njsdf_.weight[layer].transpose().col(h); //derivative wrt input
    //                 #pragma omp parallel for
    //                 for (int w = 0; w < 255; w++)
    //                 {
    //                     njsdf_.hidden_derivative[layer].coeffRef(w, h) = njsdf_.weight[layer].coeff(w,h);
    //                 }
    //             }
    //         }
    //         // njsdf_.hidden_derivative[layer] = njsdf_.hidden_derivative[layer].transpose();
    //         finish[layer*3+1] = clock();

    //         start[3*layer+2] = clock(); //Multip
    //         temp_derivative =  njsdf_.hidden_derivative[layer] * temp_derivative;
    //         finish[3*layer+2] = clock();
    //     }
    // }
    // finish[3*njsdf_.n_layer - 1] = clock();
    // if(time_verbose)
    // {
    //     std::cout<<"------------------Time[1e-6]------------------"<<std::endl;
    //     for (int layer = 0; layer < njsdf_.n_layer; layer++)
    //     {
    //         if(layer == njsdf_.n_layer - 1)
    //         {
    //             std::cout<<"Layer "<<layer<<" -Linear: "<<double(finish[3*layer+0]-start[3*layer+0])<<std::endl;
    //             std::cout<<"Layer "<<layer<<" -Multip: "<<double(finish[3*layer+1]-start[3*layer+1])<<std::endl;
    //             std::cout<<"Total          : "          <<double(finish[3*layer+2]-start[3*layer+2])<<std::endl;
    //         }
    //         else
    //         {
    //             std::cout<<"Layer "<<layer<<" -Linear: "<<double(finish[3*layer+0]-start[3*layer+0])<<std::endl;
    //             std::cout<<"Layer "<<layer<<" -ReLU  : "<<double(finish[3*layer+1]-start[3*layer+1])<<std::endl;
    //             std::cout<<"Layer "<<layer<<" -Multip: "<<double(finish[3*layer+2]-start[3*layer+2])<<std::endl;
    //         }
    //     }
    //     std::cout<<"------------------------------------------------"<<std::endl;
    // }
    // #pragma omp parallel
    // #pragma omp for
    clock_t tic, toc;

    Eigen::VectorXd tmp;

    tic = clock();

    // #pragma omp parallel for
    for(int i = 0 ;i < 100; i++)
    {
        tmp(i) = double(i);
    }

    toc = clock();

    std::cout<<double(toc-tic)<<std::endl;

    return std::make_pair(njsdf_.output, njsdf_.output_derivative); 
}



namespace py = pybind11;

PYBIND11_MODULE(libNJSDF_FUN, m)
{
    m.def("setNeuralNetwork", &setNeuralNetwork, "setNeuralNetwork");
    m.def("setNetworkInput", &setNetworkInput, "setNetworkInput");
    m.def("calculateMlpOutput", &calculateMlpOutput, "calculateMlpOutput");
}
// g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` NJSDF.cpp -o NJSDF_FUN.so