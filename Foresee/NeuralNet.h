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
	// Desconstrutor
	~NeuralNet();

	/*
	@param[in]	float lowerBound	primeiro limitador
	@param[in]	float upperBound	segundo limitador
	@Return		float				número aleatório entre dois números
	*/
	static float getRandomBetween(float lowerBound, float upperBound);


	/*
	* @param[in]	int		rows				número de linhas
	* @param[in]	int		columns				número de colunas
	* @param[in]	float	randomLowerBound	limite inferior do número aleatório
	* @param[in]	float	randomUpperBound	limite superior do número aleatório
	* @Return vector<vector<float>>	matriz aleatória
	*/
	static std::vector< std::vector<float> > randomMatrixCreator(int rows, int columns, float randomLowerBound, float randomUpperBound);
	
	
	/*
	* @param[in]	int		rows				número de linhas
	* @param[in]	int		columns				número de colunas
	* @param[in]	float	randomLowerBound	limite inferior do número aleatório
	* @param[in]	float	randomUpperBound	limite superior do número aleatório
	* @Return vector<vector<float>>	matriz aleatória
	*/
	static std::vector<float> randomVectorCreator(int rows, float randomLowerBound, float randomUpperBound);


	/*
	* @param[in]	vector<vector<float>>	matriz
	*/
	static void printMatrix(std::vector< std::vector<float> > matrix);

	/*
	* @param[in]	vector<vector<float>>	matriz
	*/
	static void printVector(std::vector<float> row);


	/*
	* @param[in]	vector<vector<float>>	matriz
	* @param[in]	int&	linhas
	* @param[in]	int&	colunas
	*/
	static void getMatrixSize(int &rows, int &columns, std::vector< std::vector<float> > matrix);
	

	/*
	* @param[in]	vector<vector<float>>	matriz de pesos da rede
	* @param[in]	vector<float>	inputs da rede
	* return		float	saída do neurônio
	*/
	static std::vector<float> feedForward(std::vector<std::vector<float>> weightMatrix, std::vector<float> input, std::vector<float> &net);


	/*
	* @param[in]	vector<float>	matriz de pesos
	* @param[in]	vector<float>	inputs da rede
	* return		float			saída do neurônio
	*/
	static float feedForward(std::vector<float> weightVector, std::vector<float> input, float &net);

	static std::vector<float> calculateOutputNeuronError(std::vector<float> target, std::vector<float> output, std::vector<float> net);

	/*
	* Calcula o erro de um neurônio da camada escondida
	static std::vector<float> calculateHiddenNeuronError(std::vector<float> outputError, std::vector<std::vector<float>> weightMatrix, std::vector<float> net);


	/*
	 * @param[in] float	soma ponderada das entradas e pesos do neurônio
	 * @param[in] string	função a ser utilizada. Por enquanto só tanh como função de ativação disponível
	 * @Return float	saída do neurônio
	 */
	static float activationFunction(float net);
	
	
	/*
	 * @param[in] float		soma ponderada das entradas e pesos do neurônio
	 * @param[in] string		função a ser utilizada para ativação. Aqui será utilizada a derivada. 
	 *						Por enquanto só tanh como função de ativação disponível
	 * @Return float		saída do neurônio
	 */
	static float activFunDerivative(float net);


	//
	//
	///*
	// * Programa principal
	// */
	//void main();
	//
	//
	///*
	// * Programa principal
	// */
	//void main2();

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