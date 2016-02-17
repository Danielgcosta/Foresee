// NeuralNet.h	Declaração
//
// Daniel Guimarães Costa
// 28/01/2016

#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>

class NeuralNet {
private:

public:
	// Construtor
	NeuralNet();
	// Destrutor
	~NeuralNet();

	/*
	Retorna um número aleatório entre lowerBound e upperBound.
	@param[in]	primeiro limitador
	@param[in]	segundo limitador
	@return[out]	número aleatório entre dois números
	*/
	static float getRandomBetween(float lowerBound, float upperBound);


	/*
	Gera uma matriz aleatória.
	@param[in]	número de linhas
	@param[in]	número de colunas
	@param[in]	limite inferior do número aleatório
	@param[in]	limite superior do número aleatório
	@return[out] matriz aleatória
	*/
	static std::vector< std::vector<float> > randomMatrixCreator(int rows, int columns, float randomLowerBound, float randomUpperBound);
	
	
	/*
	Gera um vetor aleatório.
	@param[in]	número de linhas
	@param[in]	número de colunas
	@param[in]	limite inferior do número aleatório
	@param[in]	limite superior do número aleatório
	@return[out] vetor aleatório
	*/
	static std::vector<float> randomVectorCreator(int rows, float randomLowerBound, float randomUpperBound);


	/*
	Imprime a matriz no terminal.
	@param[in]	matriz
	*/
	static void printMatrix(std::vector< std::vector<float> > matrix);

	/*
	Imprime o vetor no terminal.
	@param[in]	vetor
	*/
	static void printVector(std::vector<float> row);


	/*
	Retorna o tamanho da matriz.
	@param[in]	matriz
	@param[in]	& linhas
	@param[in]	& colunas
	*/
	static void getMatrixSize(int &rows, int &columns, std::vector< std::vector<float> > matrix);
	

	/*
	Adiciona o vetor como linha na matrix.
	@param[in]	vetor
	@param[in]	& matriz
	*/
	static void addRowToMatrix(std::vector<float>, std::vector< std::vector<float> > &);


	/*
	Multiplica duas matrizes.
	@param[in]	matriz
	@param[in]	matriz
	@return[out] matriz
	*/
	static std::vector<std::vector<float>> multiplyMatrix(std::vector<std::vector<float>> matrix_1, std::vector<std::vector<float>>matrix_2);

	static std::vector<std::vector<float>> multiplyMatrix(std::vector<float> matrix_1, std::vector<std::vector<float>> matrix_2);

	static std::vector<std::vector<float>> multiplyMatrix(std::vector<std::vector<float>> matrix_1, std::vector<float> matrix_2);

	/*
	Soma um vetor a outro, elemento a elemento
	*/
	static void sumToVector(std::vector<float> &row_1, std::vector<float> row_2);

	static void sumToVector(std::vector<float> &row, float value);

	/*
	Calcula a NET de todos os neurônios da camada.
	@param[in]	matriz de pesos da camada
	@param[in]	inputs da camada
	@return[out]	saída da camada
	*/
	static std::vector<float> feedForward(std::vector<float> input, std::vector<std::vector<float>> weightMatrix);


	/*
	Calcula a NET de todos os neurônios da camada.
	@param[in]	matriz de pesos da camada
	@param[in]	inputs da camada
	@return[out]	saída da camada
	*/
	static float feedForward(std::vector<float> input, std::vector<float> weightVector);


	/*
	Calcula o erro do neurônio de saída.
	@param[in]	objetivo do neurônio
	@param[in]	saída do neurônio
	@param[in]	net da camada de saída
	@return[out]	erro do neurônio de saída
	*/
	static float calculateOutputNeuronError(float target, float output, float net);

	static std::vector<float> calculateOutputNeuronError(std::vector<float> target, std::vector<float> output, std::vector<float> net);

	/*
	Calcula os erros dos neurônios da camada escondida.
	@param[in]	vetor de erros dos neurônios da camada seguinte
	@param[in]	matriz de pesos da camada
	@param[in]	net da camada escondida de origem
	@return[out]	erro do neurônio da camada escondida de origem
	*/
	static std::vector<float> NeuralNet::calculateHiddenNeuronError(float outputError, std::vector<std::vector<float>> weightMatrix, std::vector<float> net);

	static std::vector<float> calculateHiddenNeuronError(std::vector<float> outputError, std::vector<std::vector<float>> weightMatrix, std::vector<float> net);


	/*
	Corrije as matrizes de pesos.
	@param[in]	matrix de pesos a corrigir
	@param[in]	taxa de aprendizado
	@param[in]	erro do(s) neurônio(s) 
	@param[in]	entrada do(s) neurônio(s)
	*/
	static void adjustWeightMatrix(std::vector<float> &weightMatrix, float learningRate, float neuronError, std::vector<float>input);
	
	static void adjustWeightMatrix(std::vector<std::vector<float>> &weightMatrix, float learningRate, std::vector<float> neuronError, std::vector<float>input);

	static void adjustWeightMatrix(std::vector<std::vector<float>> &weightMatrix, float learningRate, float neuronError, std::vector<float>input);

	/*
	Organiza os dados temporais em uma matriz. 
	Cada linha é uma amostra para treinamento
	*/
	static std::vector<std::vector<float>> prepareTimeSeriesForInput(std::vector<float> timeData, int windowSize, std::vector<float> &target);
	

	/*
	Função de ativação dos neurônios
	@param[in] float	soma ponderada das entradas e pesos do neurônio	 
		TODO: Por enquanto só tanh como função de ativação disponível.
			@param[in] string	função a ser utilizada.
			Possibilitar entrada de texto ex.: "tanh" para escolher a função de ativação
	@return[out] float	saída do neurônio
	*/
	static float activationFunction(float net);

	static std::vector<float> activationFunction(std::vector<float> net);
	
	
	/*
	Derivada da função de ativação
	@param[in] float		soma ponderada das entradas e pesos do neurônio
		TODO: Por enquanto só tem a derivada de tanh como função de ativação disponível. 
			@param[in] string		função a ser utilizada para ativação. Aqui será utilizada a derivada.
			Possibilitar entrada de texto ex.: "tanh" para escolher a função de ativação
	@return[out] float		saída do neurônio
	 */
	static float activFunDerivative(float net);
	

	// tamanho máximo de array aleatório
	int _sizeLimit = 250000;
	// taxa de aprendizado
	float _learnRate = 0.3f;
	// tolerância 
	float _tolerance = .01f;
	// tamanho da janela
	int _windowSize = 5;
};
#endif NEURALNET_H