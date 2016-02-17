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
//TODO: Criar classe Neuron e tirar static dos métodos

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
	getMatrixSize(rows, columns, matrix);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			printf("%1.6f\t", matrix.at(i).at(j));
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

void NeuralNet::printVector(std::vector<float> row){
	for (unsigned int i = 0; i < row.size(); i++) {
		printf("%1.6f\n", row.at(i));		
	}
	std::cout << "\n";
}

void NeuralNet::getMatrixSize(int &rows, int &columns, std::vector< std::vector<float> > matrix){
	rows = int(matrix.size());
	columns = int(matrix.at(0).size());
}

void NeuralNet::addRowToMatrix(std::vector<float> row, std::vector< std::vector<float> > &matrix) {
	matrix.push_back(row);	
	return;
}

vector<vector<float>> NeuralNet::multiplyMatrix(vector<vector<float>> matrix_1, vector<vector<float>> matrix_2){
	vector<vector<float>> resultMatrix;
	int rows_1;
	int columns_1;
	int rows_2;
	int columns_2;
	NeuralNet::getMatrixSize(rows_1, columns_1, matrix_1);
	NeuralNet::getMatrixSize(rows_2, columns_2, matrix_2);
	if (columns_1 != rows_2) {
		printf("\n%s\n", "As matrizes não possuem tamanhos compatíveis");
		printf("%s %i %s %i %s\n", "Matriz 1: ", rows_1, "linhas e", columns_1, "colunas.");
		printf("%s %i %s %i %s\n", "Matriz 2: ", rows_2, "linhas e", columns_2, "colunas.");
	}
	else {
		for ( int nRow = 0; nRow < rows_1; nRow++) {
			vector<float> row;
			for ( int nColumn = 0; nColumn < columns_2; nColumn++) {
				float element = 0;
				for ( int k = 0; k < columns_1; k++) {
					element = element+ matrix_1.at(nRow).at(k)*matrix_2.at(k).at(nColumn);
				}
				row.push_back(element);
			}
			resultMatrix.push_back(row);
		}
	}
	return resultMatrix;
}

vector<vector<float>> NeuralNet::multiplyMatrix(vector<float> matrix_1, vector<vector<float>> matrix_2) {
	vector<vector<float>> matrix;
	matrix.push_back(matrix_1);
	return NeuralNet::multiplyMatrix(matrix, matrix_2);
}

vector<vector<float>> NeuralNet::multiplyMatrix(vector<vector<float>> matrix_1, vector<float> matrix_2) {
	vector<vector<float>> matrix;
	matrix.push_back(matrix_2);
	return NeuralNet::multiplyMatrix(matrix_1, matrix);
}

void NeuralNet::sumToVector(vector<float> &row_1, vector<float> row_2) {
	if (row_1.size() == row_2.size()) {
		for (unsigned int element = 0; element < row_1.size(); element++) {
			row_1.at(element) = row_1.at(element) + row_2.at(element);
		}
	}
	else {
		printf("%s", "Vetores de tamanhos incompatíveis");
	}
}

void NeuralNet::sumToVector(vector<float> &row, float value) {
	for (unsigned int element = 0; element < row.size(); element++) {
		row.at(element) = row.at(element) + value;
	}
}

vector<float> NeuralNet::feedForward(vector<float> input, vector<vector<float>> weightMatrix){
	vector<vector<float>> inputMatrix;
	inputMatrix.push_back(input);
	vector<vector<float>> net = multiplyMatrix(inputMatrix, weightMatrix);
	return net.at(0);
}

float NeuralNet::feedForward(vector<float> input, vector<float> weightVector) {
	vector<vector<float>> inputMatrix;
	inputMatrix.push_back(input);
	vector<vector<float>> weightMatrix;
	inputMatrix.push_back(weightVector);
	vector<vector<float>> net = multiplyMatrix(inputMatrix, weightMatrix);
	return net.at(0).at(0);
}

float NeuralNet::calculateOutputNeuronError(float target, float output, float net) {
	return (target - output) * NeuralNet::activFunDerivative(net);
}

vector<float> NeuralNet::calculateOutputNeuronError(vector<float> target, vector<float> output, vector<float> net) {
	vector<float> error;
	for (unsigned int i = 0; i < target.size(); i++) {
		error.push_back(NeuralNet::calculateOutputNeuronError(target.at(i), output.at(i), net.at(i)));
	}
	return error;
}

vector<float> NeuralNet::calculateHiddenNeuronError(float outputError, vector<vector<float>> weightMatrix, vector<float> net) {
	vector<float> error;
	for (unsigned int neuron = 0; neuron < net.size(); neuron++) {
		error.push_back(outputError*weightMatrix.at(neuron).at(0)* NeuralNet::activFunDerivative(net.at(neuron)));
	}
	return error;
}

vector<float> NeuralNet::calculateHiddenNeuronError(vector<float> outputError, vector<vector<float>> weightMatrix, vector<float> net) {
	vector<float> error;
	vector<vector<float>> matrix = NeuralNet::multiplyMatrix(outputError, weightMatrix);
	for (unsigned int neuron = 0; neuron < net.size(); neuron++) {
		error.push_back(matrix.at(0).at(neuron)* NeuralNet::activFunDerivative(net.at(neuron)));
	}
	return error;
}	

void NeuralNet::adjustWeightMatrix(vector<float> &weightMatrix, float learningRate, float neuronError, vector<float>input) {
	for (unsigned int i = 0; i < input.size(); i++) {
		weightMatrix.at(i) = weightMatrix.at(i)+learningRate*neuronError*input.at(i);
	}
}

void NeuralNet::adjustWeightMatrix(vector<vector<float>> &weightMatrix, float learningRate, vector<float> neuronError, vector<float>input) {
	for (unsigned int i = 0; i < input.size(); i++) {
		for (int neuron = 0; neuron < neuronError.size(); neuron++) {
			weightMatrix.at(i).at(neuron) = weightMatrix.at(i).at(neuron)+learningRate*neuronError.at(neuron)*input.at(i);
			printf("%s %f\n", "correção: ", learningRate*neuronError.at(neuron)*input.at(i));
			printf("%s %f\n", "resultado: ", weightMatrix.at(i).at(neuron));
		}
	}
}

void NeuralNet::adjustWeightMatrix(vector<vector<float>> &weightMatrix, float learningRate, float neuronError, vector<float>input) {
	vector<vector<float>> matrix;
	for (unsigned int i = 0; i < input.size(); i++) {
		for (int neuron = 0; neuron < weightMatrix.at(i).size(); neuron++) {
			weightMatrix.at(i).at(neuron) = weightMatrix.at(i).at(neuron)+learningRate*neuronError*input.at(i);
			printf("%s %f\n", "correção: ", learningRate*neuronError*input.at(i));
			printf("%s %f\n", "resultado: ", weightMatrix.at(i).at(neuron));
		}
	}
	printf("\n");
}


vector<vector<float>> NeuralNet::prepareTimeSeriesForInput(vector<float> timeData, int windowSize, vector<float> &target) {
	vector<vector<float>> inputDataMatrix;
	for (unsigned int i = 0; (i < timeData.size() - windowSize); i++)
	{
		// Criação dos subconjuntos de input e target
		vector<float> input;
		for (unsigned int j = i; j < windowSize + i; j++) {
			// Criação do target
			input.push_back(timeData.at(j));
		}
		// Input para ativar o bias
		input.push_back(int(1));
		target.push_back(timeData.at(i + windowSize));
		inputDataMatrix.push_back(input);
	}
	return inputDataMatrix;
}

float NeuralNet::activationFunction(float net){
	float result = tanh(net);
	return result;
}

vector<float> NeuralNet::activationFunction(vector<float> net){
	vector<float> output;
	for (unsigned int i = 0; i<net.size(); i++) {
		output.push_back(tanh(net.at(i)));
	}
	return output;
}

float NeuralNet::activFunDerivative(float net)
{
	float result = 1 / (pow(cosh(net), 2));	//sech²(net);
	return result;
}
