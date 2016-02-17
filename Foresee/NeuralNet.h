// NeuralNet.h	Declara��o
//
// Daniel Guimar�es Costa
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
	Retorna um n�mero aleat�rio entre lowerBound e upperBound.
	@param[in]	primeiro limitador
	@param[in]	segundo limitador
	@return[out]	n�mero aleat�rio entre dois n�meros
	*/
	static float getRandomBetween(float lowerBound, float upperBound);


	/*
	Gera uma matriz aleat�ria.
	@param[in]	n�mero de linhas
	@param[in]	n�mero de colunas
	@param[in]	limite inferior do n�mero aleat�rio
	@param[in]	limite superior do n�mero aleat�rio
	@return[out] matriz aleat�ria
	*/
	static std::vector< std::vector<float> > randomMatrixCreator(int rows, int columns, float randomLowerBound, float randomUpperBound);
	
	
	/*
	Gera um vetor aleat�rio.
	@param[in]	n�mero de linhas
	@param[in]	n�mero de colunas
	@param[in]	limite inferior do n�mero aleat�rio
	@param[in]	limite superior do n�mero aleat�rio
	@return[out] vetor aleat�rio
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
	Calcula a NET de todos os neur�nios da camada.
	@param[in]	matriz de pesos da camada
	@param[in]	inputs da camada
	@return[out]	sa�da da camada
	*/
	static std::vector<float> feedForward(std::vector<float> input, std::vector<std::vector<float>> weightMatrix);


	/*
	Calcula a NET de todos os neur�nios da camada.
	@param[in]	matriz de pesos da camada
	@param[in]	inputs da camada
	@return[out]	sa�da da camada
	*/
	static float feedForward(std::vector<float> input, std::vector<float> weightVector);


	/*
	Calcula o erro do neur�nio de sa�da.
	@param[in]	objetivo do neur�nio
	@param[in]	sa�da do neur�nio
	@param[in]	net da camada de sa�da
	@return[out]	erro do neur�nio de sa�da
	*/
	static float calculateOutputNeuronError(float target, float output, float net);

	static std::vector<float> calculateOutputNeuronError(std::vector<float> target, std::vector<float> output, std::vector<float> net);

	/*
	Calcula os erros dos neur�nios da camada escondida.
	@param[in]	vetor de erros dos neur�nios da camada seguinte
	@param[in]	matriz de pesos da camada
	@param[in]	net da camada escondida de origem
	@return[out]	erro do neur�nio da camada escondida de origem
	*/
	static std::vector<float> NeuralNet::calculateHiddenNeuronError(float outputError, std::vector<std::vector<float>> weightMatrix, std::vector<float> net);

	static std::vector<float> calculateHiddenNeuronError(std::vector<float> outputError, std::vector<std::vector<float>> weightMatrix, std::vector<float> net);


	/*
	Corrije as matrizes de pesos.
	@param[in]	matrix de pesos a corrigir
	@param[in]	taxa de aprendizado
	@param[in]	erro do(s) neur�nio(s) 
	@param[in]	entrada do(s) neur�nio(s)
	*/
	static void adjustWeightMatrix(std::vector<float> &weightMatrix, float learningRate, float neuronError, std::vector<float>input);
	
	static void adjustWeightMatrix(std::vector<std::vector<float>> &weightMatrix, float learningRate, std::vector<float> neuronError, std::vector<float>input);

	static void adjustWeightMatrix(std::vector<std::vector<float>> &weightMatrix, float learningRate, float neuronError, std::vector<float>input);

	/*
	Organiza os dados temporais em uma matriz. 
	Cada linha � uma amostra para treinamento
	*/
	static std::vector<std::vector<float>> prepareTimeSeriesForInput(std::vector<float> timeData, int windowSize, std::vector<float> &target);
	

	/*
	Fun��o de ativa��o dos neur�nios
	@param[in] float	soma ponderada das entradas e pesos do neur�nio	 
		TODO: Por enquanto s� tanh como fun��o de ativa��o dispon�vel.
			@param[in] string	fun��o a ser utilizada.
			Possibilitar entrada de texto ex.: "tanh" para escolher a fun��o de ativa��o
	@return[out] float	sa�da do neur�nio
	*/
	static float activationFunction(float net);

	static std::vector<float> activationFunction(std::vector<float> net);
	
	
	/*
	Derivada da fun��o de ativa��o
	@param[in] float		soma ponderada das entradas e pesos do neur�nio
		TODO: Por enquanto s� tem a derivada de tanh como fun��o de ativa��o dispon�vel. 
			@param[in] string		fun��o a ser utilizada para ativa��o. Aqui ser� utilizada a derivada.
			Possibilitar entrada de texto ex.: "tanh" para escolher a fun��o de ativa��o
	@return[out] float		sa�da do neur�nio
	 */
	static float activFunDerivative(float net);
	

	// tamanho m�ximo de array aleat�rio
	int _sizeLimit = 250000;
	// taxa de aprendizado
	float _learnRate = 0.3f;
	// toler�ncia 
	float _tolerance = .01f;
	// tamanho da janela
	int _windowSize = 5;
};
#endif NEURALNET_H