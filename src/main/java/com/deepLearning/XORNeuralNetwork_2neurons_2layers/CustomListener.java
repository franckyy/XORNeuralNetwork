package com.deepLearning.XORNeuralNetwork_2neurons_2layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;

public class CustomListener implements TrainingListener{
	private final List<Double> scoreHistory = new ArrayList<>();
	private int iterationCount = 0;
	
    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
    	if (iterationCount % 10 == 0) { // Enregistrer toutes les 10 itérations
            double score = model.score();
            scoreHistory.add(score);
            System.out.println("Epoch " + epoch + ", Iteration " + iteration + " - Score: " + score);
        }
    	iterationCount++;
    }

    public List<Double> getScoreHistory() {
        return scoreHistory;
    }

    @Override
    public void onEpochStart(Model model) {}
    @Override
    public void onEpochEnd(Model model) {}
    @Override
    public void onForwardPass(Model model, List<INDArray> activations) {}
    @Override
    public void onBackwardPass(Model model) {}

	@Override
	public void onGradientCalculation(Model model) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void onForwardPass(Model model, Map<String, INDArray> activations) {
		// TODO Auto-generated method stub
		
	}
}
