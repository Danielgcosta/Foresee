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
	// Desconstrutor
	~NeuralNet();

	/*
	@param[in]	float lowerBound	primeiro limitador
	@param[in]	float upperBound	segundo limitador
	@Return		float				n�mero aleat�rio entre dois n�meros
	*/
	static float getRandomBetween(float lowerBound, float upperBound);


	/*
	* @param[in]	int		rows				n�mero de linhas
	* @param[in]	int		columns				n�mero de colunas
	* @param[in]	float	randomLowerBound	limite inferior do n�mero aleat�rio
	* @param[in]	float	randomUpperBound	limite superior do n�mero aleat�rio
	* @Return vector<vector<float>>	matriz aleat�ria
	*/
	static std::vector< std::vector<float> > randomMatrixCreator(int rows, int columns, float randomLowerBound, float randomUpperBound);
	
	
	/*
	* @param[in]	int		rows				n�mero de linhas
	* @param[in]	int		columns				n�mero de colunas
	* @param[in]	float	randomLowerBound	limite inferior do n�mero aleat�rio
	* @param[in]	float	randomUpperBound	limite superior do n�mero aleat�rio
	* @Return vector<vector<float>>	matriz aleat�ria
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
	* return		float	sa�da do neur�nio
	*/
	static std::vector<float> feedForward(std::vector<std::vector<float>> weightMatrix, std::vector<float> input, std::vector<float> &net);


	/*
	* @param[in]	vector<float>	matriz de pesos
	* @param[in]	vector<float>	inputs da rede
	* return		float			sa�da do neur�nio
	*/
	static float feedForward(std::vector<float> weightVector, std::vector<float> input, float &net);

	static std::vector<float> calculateOutputNeuronError(std::vector<float> target, std::vector<float> output, std::vector<float> net);

	/*
	* Calcula o erro de um neur�nio da camada escondida
	static std::vector<float> calculateHiddenNeuronError(std::vector<float> outputError, std::vector<std::vector<float>> weightMatrix, std::vector<float> net);


	/*
	 * @param[in] float	soma ponderada das entradas e pesos do neur�nio
	 * @param[in] string	fun��o a ser utilizada. Por enquanto s� tanh como fun��o de ativa��o dispon�vel
	 * @Return float	sa�da do neur�nio
	 */
	static float activationFunction(float net);
	
	
	/*
	 * @param[in] float		soma ponderada das entradas e pesos do neur�nio
	 * @param[in] string		fun��o a ser utilizada para ativa��o. Aqui ser� utilizada a derivada. 
	 *						Por enquanto s� tanh como fun��o de ativa��o dispon�vel
	 * @Return float		sa�da do neur�nio
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