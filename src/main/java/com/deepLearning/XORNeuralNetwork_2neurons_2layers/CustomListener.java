package com.deepLearning.XORNeuralNetwork_2neurons_2layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

public class CustomListener implements TrainingListener{
	 private List<Double> scoreHistory = new ArrayList<>();

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        double score = model.score();
        scoreHistory.add(score);
        System.out.println("Epoch " + epoch + " | Iteration " + iteration + " | Score: " + score);
    }

    public List<Double> getScoreHistory() {
        return scoreHistory;
    }

	@Override
	public void onEpochStart(Model model) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void onEpochEnd(Model model) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void onForwardPass(Model model, List<INDArray> activations) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void onForwardPass(Model model, Map<String, INDArray> activations) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void onGradientCalculation(Model model) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void onBackwardPass(Model model) {
		// TODO Auto-generated method stub
		
	}
}
