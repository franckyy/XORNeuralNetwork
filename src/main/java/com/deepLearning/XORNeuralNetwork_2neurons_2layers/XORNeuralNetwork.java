package com.deepLearning.XORNeuralNetwork_2neurons_2layers;

import java.util.Collections;
import java.util.List;

import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class XORNeuralNetwork {
    public static void main(String[] args) {

        // 1. Données d'entraînement (XOR)
        INDArray input = Nd4j.create(new double[][]{
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        });

        INDArray labels = Nd4j.create(new double[][]{
                {0}, // 0 XOR 0 = 0
                {1}, // 0 XOR 1 = 1
                {1}, // 1 XOR 0 = 1
                {0}  // 1 XOR 1 = 0
        });

        DataSet dataSet = new DataSet(input, labels);
        List<DataSet> list = dataSet.asList();
        Collections.shuffle(list);
        DataSetIterator iterator = new ListDataSetIterator<>(list, 4);

        // 2. Définition du modèle
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123) // Pour la reproductibilité
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new org.nd4j.linalg.learning.config.Sgd(0.5)) // Taux d'apprentissage
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(2) // 2 entrées
                        .nOut(8) // 2 neurones dans la première couche cachée
                        .activation(Activation.TANH) // Fonction d'activation ReLU
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(2) // 2 entrées (de la couche précédente)
                        .nOut(8) // 2 neurones dans la deuxième couche cachée
                        .activation(Activation.TANH) // ReLU encore
                        .build())
                .layer(2, new OutputLayer.Builder()
                        .nIn(8) // 2 entrées (de la dernière couche cachée)
                        .nOut(1) // 1 sortie
                        .activation(Activation.SIGMOID) // Sigmoïde pour classification binaire
                        .lossFunction(LossFunctions.LossFunction.XENT) // Entropie croisée
                        .build())
                .build();

        // 3. Initialisation et entraînement du modèle
        org.deeplearning4j.nn.multilayer.MultiLayerNetwork model = new org.deeplearning4j.nn.multilayer.MultiLayerNetwork(config);
        model.init();

    	CustomListener listener = new CustomListener();
    	model.setListeners(listener);

        int epochs = 20000; // Nombre d'itérations d'entraînement
        for (int i = 0; i < epochs; i++) {
            iterator.reset();
            model.fit(iterator);
        }

        // 4. Test du modèle
        System.out.println("\n=== TEST DU MODÈLE ===");
        INDArray testInput = input;
        INDArray output = model.output(testInput);
        System.out.println(output);

        // 5. Affichage des résultats
        System.out.println("\nRésultats attendus :");
        System.out.println(labels);
        System.out.println("\nPrédictions du modèle :");
        System.out.println(output);
    }
}

