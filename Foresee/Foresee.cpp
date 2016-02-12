/*
Foresee
Algoritmo de Redes Neurais para previsão de séries em C++

28/01/2016
Daniel Guimarães Costa
dcosta@ele.puc-rio.br
dcosta@tecgraf.puc-rio.br
*/

#include "stdafx.h"
#include "NeuralNet.h"
#include "iostream"	// impressão/obtenção de dados do/para usuário
using namespace std;

int main()
{
	// Entrada
	vector<float> timeData { 1,2,3,4,5,6,7,8,9,10,11 };	// Série temporal de exemplo

	int nInputs = 4;    //# de elementos da janela
	int nNeuronios = 6;	//# neurônios da camada escondida
	//int nSaida = 1;	//# neurônios da camada de saída
	float randomNumberLimit = float(0.2);

	// Preparando os dados para a rede
	vector<vector<float>> inputDataMatrix;	// Cada linha é uma entrada de treinamento
	vector<float> target;					// Cada elemento é um target da linha de mesmo índice
	for (unsigned int i = 0; (i  < timeData.size() - nInputs); i++)
	{
		// Criação dos subconjuntos de input e target
		vector<float> input;
		for (unsigned int j = i; j < nInputs + i; j++) {
			// Criação do target
			input.push_back(timeData.at(j));
		}
		target.push_back(timeData.at(i + nInputs));
		inputDataMatrix.push_back(input);
	}

	// Inicializa a rede neural	
	// Matriz de pesos da camada escondida
	vector<vector<float>> hiddenLayerWeightMatrix = NeuralNet::randomMatrixCreator(nInputs+1, nNeuronios, -randomNumberLimit, randomNumberLimit);
	// Mostra a matriz de pesos criada
	NeuralNet::printMatrix(hiddenLayerWeightMatrix);
	// Matriz de pesos da camada de saída
	vector<float> outputWeightMatrix = NeuralNet::randomVectorCreator(nNeuronios+1, -randomNumberLimit, randomNumberLimit);
	// Mostra a matriz de pesos criada
	NeuralNet::printVector(outputWeightMatrix);
	
	// Feed Forward
	vector<float> input;	// entradas da rede
	vector<vector<float>> hiddenLayerOutput;	// vetor com saída das camadas escondidas de cada caso
	vector<float> output;	// saída da rede para cada caso
	vector<float> hiddenLayerCaseNets;		// net da camada escondida de um caso
	vector<vector<float>> hiddenLayerNets;	// net da camada escondida de todos os casos
	float outputLayerCaseNet;		// saída de cada caso
	vector<float> outputLayerNets;	// vetor com as saídas de todos os casos
	// Roda a Feed Forward para cada caso (
	for (unsigned int i = 0; i < inputDataMatrix.size(); i++)
	{
		printf("%s %i\n", "Rodando o caso ", i + 1);
		input = inputDataMatrix.at(i);		
		hiddenLayerOutput.push_back(NeuralNet::feedForward(hiddenLayerWeightMatrix, input, hiddenLayerCaseNets));
		hiddenLayerNets.push_back(hiddenLayerCaseNets);
		output.push_back(NeuralNet::feedForward(outputWeightMatrix, hiddenLayerOutput.at(i), outputLayerCaseNet));
		outputLayerNets.push_back(outputLayerCaseNet);
	}

	// Back Propagation
	float learningRate = 0.5;	
	// Erro dos neurônios da camada de saída
	vector<float> outputError = NeuralNet::calculateOutputNeuronError(target, output, outputLayerNets);
	// Erro dos neurônios da camada escondida
	vector<float> hiddenLayerError = NeuralNet::calculateHiddenNeuronError(outputError, hiddenLayerWeightMatrix, hiddenLayerNets);

	int i;
	cin >> i;
    return 1;
}
