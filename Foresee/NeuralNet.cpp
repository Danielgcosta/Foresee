// NeuralNet.cpp	Definição 
//
// Daniel Guimarães Costa
// 28/01/2016

#include "stdafx.h"
#include "cstdio"	// C Standard Input and Output Library
#include "string"
#include <vector>
#include "NeuralNet.h"
#include "iostream"

using namespace std;

NeuralNet::NeuralNet(){
}

NeuralNet::~NeuralNet(){
}

float NeuralNet::getRandomBetween(float lowerBound, float upperBound)
{
	float randomNumber = 2 * static_cast<float>(rand() / (pow(2, 15) - 1)) - 1;			// Gera aleatório em [ -1 ; 1 ]
	return ((upperBound - lowerBound)*randomNumber + (lowerBound + upperBound)) / 2;	// Transforma para [ limit_a ; limit_b ]
}

vector< vector<float> > NeuralNet::randomMatrixCreator(int rows, int columns, float randomLowerBound, float randomUpperBound)
{
	vector< vector<float> > matrix;
	for (int i = 0; i < rows; i++)
	{
		vector<float> row;
		for (int j = 0; j < columns; j++)
		{
			row.push_back(getRandomBetween(randomLowerBound, randomUpperBound));
		}
		matrix.push_back(row);
	}
	return matrix;
}


vector<float> NeuralNet::randomVectorCreator(int rows, float randomLowerBound, float randomUpperBound)
{
	vector<float> row;
	for (int i = 0; i < rows; i++)
	{
		row.push_back(getRandomBetween(randomLowerBound, randomUpperBound));
	}
	return row;
}

void NeuralNet::printMatrix(std::vector< std::vector<float> > matrix)
{
	int rows;
	int columns;
	printf("%s\n", "Matriz de pesos da rede neural. linhas/colunas = entradas/saídas");
	getMatrixSize(rows, columns, matrix);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			printf("%1.6f\t", matrix.at(i).at(j));
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

void NeuralNet::printVector(std::vector<float> row)
{
	printf("%s\n", "Matriz de pesos da rede neural. linhas/colunas = entradas/saídas");
	for (int i = 0; i < row.size(); i++) {
		printf("%1.6f\n", row.at(i));		
	}
	std::cout << "\n";
}

void NeuralNet::getMatrixSize(int &rows, int &columns, std::vector< std::vector<float> > matrix)
{
	rows = int(matrix.size());
	columns = int(matrix.at(0).size());
}

vector<float> NeuralNet::feedForward(vector<vector<float>> weightMatrix, vector<float> input, vector<float> &net){
	vector<float> output;
	float out;
	//int rows = weightMatrix.size();
	//int columns = weightMatrix.at(0).size();
	//float wij = weightMatrix.at(i).at(j);
	for (unsigned int j = 0; j < weightMatrix.at(0).size(); j++)	{		
		// Para cada entrada, associa seu peso e soma
		float sum = 0;
		printf("%s %i\n", "Neuronio", j + 1);
		for (unsigned int i = 0; i <= weightMatrix.size()-1; i++) {			
			printf("%s %i ", "Entrada", i + 1);
			if (i != weightMatrix.size()-1) {
				// wij * input(i)
				sum = sum + weightMatrix.at(i).at(j)*input.at(i);
				printf("%s %1.6f ", "=", input.at(i));				
			}
			else {
				// bias
				sum = sum + weightMatrix.at(i).at(j);
				net.push_back(sum);
				out = activationFunction(sum);
				output.push_back(out);
				printf("%s %i %s ", "=", 1,"(bias)");
			}
			printf("%s %1.4f ", "Peso", weightMatrix.at(i).at(j));
			printf("%s %1.4f\n", "Sum ", sum);
		}
		printf("%s %1.4f\n\n", "Output = f(sum) =", out);
	}
	return output;
}

float NeuralNet::feedForward(vector<float> weightVector, vector<float> input, float &net){
	float output;
	//int rows = weightMatrix.size();	
	// Para cada entrada, associa seu peso e soma
	float sum = 0;
	printf("%s\n", "Neuronio de saída");
	for (unsigned int i = 0; i <= weightVector.size() - 1; i++) {
		printf("%s %i ", "Entrada", i + 1);
		if (i != weightVector.size() - 1) {
			// wij * input(i)
			sum = sum + weightVector.at(i)*input.at(i);
			printf("%s %1.6f ", "=", input.at(i));
		}
		else {
			// bias
			net = sum + weightVector.at(i);	
			output = activationFunction(net);
			printf("%s %i %s ", "=", 1, "(bias)");
		}
		printf("%s %1.4f ", "Peso", weightVector.at(i));
		printf("%s %1.4f\n", "Sum ", net);
	} 
	printf("%s %1.4f\n\n", "Output = f(sum) =", output);
	return output;
}


vector<float> NeuralNet::calculateOutputNeuronError(vector<float> target, vector<float> output, vector<float> net) {
	vector<float> outputLayerError;
	for (int i = 0; i < target.size(); i++) {
		// erro = ( t - y ) * F'
		outputLayerError.push_back((target.at(i) - output.at(i)) * NeuralNet::activFunDerivative(net.at(i)));
	}
	return outputLayerError;
}


vector<float> NeuralNet::calculateHiddenNeuronError(vector<float> outputError, vector<vector<float>> weightMatrix, vector<float> net) {
// Para cada entrada, associa o erro da saída ao peso
	vector<float> error;
	for (unsigned int j = 0; j < weightMatrix.at(0).size(); j++) {
		float sum = 0;
		for (unsigned int i = 0; i < weightMatrix.size(); i++) {
			if (i != weightMatrix.size() - 1) {
				// wij * error(i)
				sum = sum + weightMatrix.at(i).at(j)*outputError.at(i);
				printf("%s %1.6f ", "=", outputError.at(i));
			}
			else {
				// bias
				sum = sum + weightMatrix.at(i).at(j);
			}
		}
		error.push_back(sum * activFunDerivative(net.at(j)));
	}
	return error;
}


float NeuralNet::activationFunction(float net)
{
	float result = tanh(net);
	return result;
}

float NeuralNet::activFunDerivative(float net)
{
	float result = 1 / (pow(cosh(net), 2));	//sech²(net);
	return result;
}

//	// Inicialização dos pesos	
//	// Pesos das camadas escondidas
//	// Exemplo: 
//	// 1 camada escondida, 3 neurônios nas camadas escondidas e 1 na de saída
//	// peso, peso, peso, bias
//
//
//	float sqrDifError = static_cast<float>(INFINITY);
//	while (sqrDifError > TOLERANCE)
//	{
//		for (int iter = 0; WINDOW + iter < sizeof(_input) / sizeof(float); iter++)
//		{
//			//DELETED HERE
//
//			//FEEDFORWARD
//
//			// Somas ponderadas somadas ao bias
//			// Net_j = Soma( w_i_j * x_i ) + bias_j
//			float net[] = { 0,0,0 };
//
//			// net[0] = _w1[0] * _input[0] + _w1[1] * _input[1] + _w1[2] * _input[2] + _w1[3];
//			for (unsigned int i = 0; i < 3; i++)
//			{
//				float sum = 0;
//				for (unsigned int j = 0; j < 3; j++)
//				{
//					sum = sum + _hiddenLayerWeights[i][j] * _input[j + iter];
//				}
//				net[i] = sum + _hiddenLayerWeights[i][3];
//			}
//
//			// Ativações dos neurônios
//			for (unsigned int i = 0; i < 3; i++)
//			{
//				S[i] = activationFunction(net[i]);
//			}
//
//			// Saída da rede
//			float sum = 0;
//			for (unsigned int i = 0; i < 3; i++)
//			{
//				sum = sum + _outputLayerWeights[i] * S[i];
//			}
//			float net_out = sum + _outputLayerWeights[3];
//			float output = activationFunction(net_out);
//
//
//			// Erro dos pesos da camada de saída
//			// e_j = ( target - output ) * F'((net_j)
//			float outputError = (_input[WINDOW + iter] - output) * activFunDerivative(net_out);
//			for (unsigned int i = 0; i < 3; i++)
//			{
//				_outputLayerWeightAdjustment[i] = LEARNRATE * outputError * S[i];
//			}
//			_outputLayerWeightAdjustment[3] = LEARNRATE * outputError * _outputLayerWeights[3];
//
//			// Erro dos pesos da camada escondida
//			// e_j = ( sum( e_k * w_k_j ) * F'(net_j)
//			float hiddenLayerError[3] = { 0, 0, 0 };
//			for (unsigned int i = 0; i < 3; i++)
//			{
//				float sum = 0;
//				for (unsigned int j = 0; j < 4; j++)
//				{
//					sum = sum + _outputLayerWeights[j];
//				}
//				hiddenLayerError[i] = sum * activFunDerivative(net[i]);
//			}
//			for (unsigned int i = 0; i < 3; i++)
//			{
//				for (unsigned int j = 0; j < 4; j++)
//				{
//					_hiddenLayerWeightAdjustment[i][j] = LEARNRATE * _input[j] * hiddenLayerError[i];
//					//std::cout << _hiddenLayerWeightAdjustment[i][j] << std::endl;
//				}
//			}
//
//			// Acumulando os erros dos pesos para corrigir após apresentar todos os padrões
//			for (unsigned int i = 0; i < 3; i++)
//			{
//				for (unsigned int j = 0; j < 4; j++)
//				{
//					hiddenLayerWeightAcumulation[i][j] = hiddenLayerWeightAcumulation[i][j] + _hiddenLayerWeightAdjustment[i][j];
//					outputLayerWeightAcumulation[j] = outputLayerWeightAcumulation[j] + _outputLayerWeightAdjustment[j];
//				}
//			}
//			// MOSTROU UM PADRÃO
//		}
//		// MOSTROU TODOS OS PADRÕES
//
//		// Correção dos pesos 
//		for (unsigned int i = 0; i < 3; i++)
//		{
//			for (unsigned int j = 0; j < 4; j++)
//			{
//				_hiddenLayerWeights[i][j] = _hiddenLayerWeights[i][j] + hiddenLayerWeightAcumulation[i][j];
//				_outputLayerWeights[j] = outputLayerWeightAcumulation[j];
//			}
//		}
//
//		// Tabela de visualização da matrix dos pesos
//		{
//			std::cout << "\n        |  W_i1    |  W_i2   |   W_i3  |   bias_i |";
//			for (unsigned int i = 0; i < 3; i++)
//			{
//				std::cout << "\nINPUT " << i + 1 << " | ";
//				for (unsigned int j = 0; j < 4; j++)
//				{
//					std::cout << _hiddenLayerWeights[i][j] << " | ";
//				}
//			}
//		}
//		
//		// Calculando o erro das avaliações
//		float error[30 - WINDOW];
//		for (unsigned int p = 0; p < 30 - WINDOW; p++)
//		{
//			float sum2 = 0;
//			for (unsigned int i = 0; i < 3; i++)
//			{
//				float sum3 = 0;
//				for (unsigned j = 0; j < 4; j++)
//				{
//					sum3 = sum3 + _input[j] * _hiddenLayerWeights[i][j];
//				}
//				S[i] = sum3 + _hiddenLayerWeights[i][3];
//				sum2 = sum2 + S[i] * _outputLayerWeights[i];
//			}
//			float output = sum2 + _outputLayerWeights[4];
//			error[p] = output - _input[p + WINDOW];
//			sqrDifError = sqrDifError + error[p] * error[p];
//		}
//	}
//	return;
//}
//