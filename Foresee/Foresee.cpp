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

/*
TODO: 
normalização
conjuntos de validação e teste

*/

void main() {
	//vector<float> timeData{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 };	// Série temporal de exemplo
	vector<float> timeData{ float(.05),float(.10),float(.15),float(.20),float(.25),float(.30),float(.35),float(.40),float(.45),float(.50),float(.55),float(.60),float(.60),float(.65),float(.70),float(.75),float(.80),float(.85),float(.90),float(.95),float(1.00) };	// Série temporal de exemplo
	int windowSize = 5;
	int nNeurons = 5;
	vector<float> targets;
	vector<vector<float>> inputData = NeuralNet::prepareTimeSeriesForInput(timeData, windowSize, targets);

	// Inicializa as matrizes de peso das camadas da rede
	vector<vector<float>> hiddenLayer_weightMatrix = NeuralNet::randomMatrixCreator(int(inputData.at(0).size()), nNeurons, float(-.25), float(.25));
	vector<vector<float>> outputLayer_weightMatrix = NeuralNet::randomMatrixCreator(nNeurons, int(1), float(-.25), float(.25));

	for (unsigned int epoch = 0; epoch < 50000; epoch++) {
		printf("\n%s %i\n", "Iteracao", epoch);
		// Acumularão os erros de todas as amostras
		float outputLayer_error = 0;
		vector<float> hiddenLayer_error;
		vector<float> outputLayer_errorAccumulator;		
		vector<vector<float>> hiddenLayer_outputAccumulator;
		for (unsigned int example = 0; example < inputData.size(); example++) {
			// Recupera os dados de entrada e saída para este exemplo
			vector<float> input = inputData.at(example);
			float target = targets.at(example);

			// Feed Forward de uma amostra
			vector<float> hiddenLayer_net = NeuralNet::feedForward(input, hiddenLayer_weightMatrix);
			vector<float> hiddenLayer_output = NeuralNet::activationFunction(hiddenLayer_net);
			hiddenLayer_outputAccumulator.push_back(hiddenLayer_output);
			vector<float> outputLayer_net = NeuralNet::feedForward(hiddenLayer_output, outputLayer_weightMatrix);
			vector<float> outputLayer_output = NeuralNet::activationFunction(outputLayer_net);
			// Apenas uma saída, guardada no primeiro neurônio
			float output = outputLayer_output.at(0);
			float net = outputLayer_net.at(0);

			// Cálculo dos erros para esta amostra
			float error = NeuralNet::calculateOutputNeuronError(target, output, net);
			outputLayer_errorAccumulator.push_back(error);
			NeuralNet::sumToVector(hiddenLayer_error,NeuralNet::calculateHiddenNeuronError(error, outputLayer_weightMatrix, hiddenLayer_net));
		
			// VISUALIZAÇÃO
			//printf("%s\n", "_________________________");
			//printf("%s %i\n", "exemplo", example + 1);
			//printf("%s\n", "input");
			//NeuralNet::printVector(input);
			//printf(" %s %f\n", "target", target);
			//printf(" %s %f\n", "output", output);
			//printf(" %s %f\n", "error", error);

			//printf("\n%s\n", "net da camada escondida");
			//NeuralNet::printVector(hiddenLayer_net);
			//printf("%s\n", "ativação dos neurônios da camada escondida");
			//NeuralNet::printVector(hiddenLayer_output);
			//printf("%s\n", "net da camada de saída");
			//NeuralNet::printVector(outputLayer_net);
			//printf("%s\n", "ativação dos neurônios da camada de saída");
			//NeuralNet::printVector(outputLayer_output);
			
			//printf("%s\n", "matrix de pesos da camada escondida");
			//NeuralNet::printMatrix(hiddenLayer_weightMatrix);
			//printf("%s\n", "matrix de pesos da camada de saída");
			//NeuralNet::printMatrix(outputLayer_weightMatrix);
			// VISUALIZAÇÃO

		}

		// Correção dos pesos
		//float learningRate = 0.5;
		//for (unsigned int example = 0; example < inputData.size(); example++) {
		//	NeuralNet::adjustWeightMatrix(outputLayer_weightMatrix, learningRate, outputLayer_error, hiddenLayer_outputAccumulator.at(example));
		//	NeuralNet::adjustWeightMatrix(hiddenLayer_weightMatrix, learningRate, hiddenLayer_error, inputData.at(example));
		//}
		float learningRate = 0.5;
		vector<float> outputLayer_adjustment;
		for (int input = 0; input < hiddenLayer_outputAccumulator.at(0).size(); input++) {
			float sum = 0;
			for (unsigned int example = 0; example < inputData.size(); example++) {
				sum = learningRate*outputLayer_error*hiddenLayer_outputAccumulator.at(example).at(input);
			}
			outputLayer_adjustment.push_back(sum);
		}

		vector<float> hiddenLayer_adjustment;
		for (int input = 0; input < inputData.at(0).size(); input++) {
			for (int error = 0; error < inputData.size(); error++) {
				float sum = 0;
				for (unsigned int example = 0; example < inputData.at(0).size(); example++) {
					sum = learningRate*hiddenLayer_error.at(error)*inputData.at(example).at(input);
				}
				hiddenLayer_adjustment.push_back(sum);
			}
		}

			//NeuralNet::adjustWeightMatrix(outputLayer_weightMatrix, learningRate, outputLayer_error, hiddenLayer_outputAccumulator.at(example));
			//NeuralNet::adjustWeightMatrix(hiddenLayer_weightMatrix, learningRate, hiddenLayer_error, inputData.at(example));
		
		// Testando a previsão do próximo valor
		// 16,17,18,19,20 -> 21
		vector<float> inputTest = { float(.80),float(.85),float(.90),float(.95),float(1),float(1) };
		vector<float> hiddenLayer_output = NeuralNet::activationFunction(NeuralNet::feedForward(inputTest, hiddenLayer_weightMatrix));
		vector<float> outputLayer_output = NeuralNet::activationFunction(NeuralNet::feedForward(hiddenLayer_output, outputLayer_weightMatrix));
		float output = outputLayer_output.at(0);

		printf("%s %i %s %f %s %3.2f%s\n", "iteracao: ", epoch + 1, "previsao: ", output, "erro: ", abs(output - 1.05) * 100 / 1.05, "%");
	}

	int i;
	cin >> i;
}