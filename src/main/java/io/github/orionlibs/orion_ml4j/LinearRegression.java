package io.github.orionlibs.orion_ml4j;

import io.github.orionlibs.orion_assert.Assert;
import io.github.orionlibs.orion_runnable.functions.OrionFunction1x1;
import io.github.orionlibs.orion_simple_math.statistics.Mean;
import io.github.orionlibs.orion_tuple.Pair;
import java.util.ArrayList;
import java.util.List;

public class LinearRegression
{
    //Y = b0 + b1X
    private List<Pair<Float, Float>> pairsOfInputsOutputsTrainingData;
    private List<Float> inputs;
    private List<Float> outputs;
    private float slope;
    private float intercept;
    private boolean hasRegressionRun;
    private OrionFunction1x1<Float, Float> modelFunction;


    public LinearRegression(List<Pair<Float, Float>> pairsOfInputsOutputsTrainingData)
    {
        Assert.notEmpty(pairsOfInputsOutputsTrainingData, "input cannot be empty");
        this.pairsOfInputsOutputsTrainingData = pairsOfInputsOutputsTrainingData;
        inputs = pairsOfInputsOutputsTrainingData.stream()
                        .map(pair -> pair.getFirst())
                        .toList();
        outputs = pairsOfInputsOutputsTrainingData.stream()
                        .map(pair -> pair.getSecond())
                        .toList();
    }


    public void run()
    {
        float xMean = new Mean().getMean(inputs);
        float yMean = new Mean().getMean(outputs);
        float numerator = 0.0f;
        float denominator = 0.0f;
        for(int i = 0; i < pairsOfInputsOutputsTrainingData.size(); i++)
        {
            float xMinusMean = (inputs.get(i) - xMean);
            numerator += xMinusMean * (outputs.get(i) - yMean);
            denominator += xMinusMean * xMinusMean;
        }
        slope = numerator / denominator;
        intercept = yMean - (slope * xMean);
        modelFunction = (Float x) -> intercept + (slope * x);
        hasRegressionRun = true;
    }


    public List<Float> getResiduals()
    {
        if(!hasRegressionRun)
        {
            run();
        }
        List<Float> residuals = new ArrayList<>();
        for(int i = 0; i < inputs.size(); i++)
        {
            residuals.add(outputs.get(i) - modelFunction.run(inputs.get(i)));
        }
        return residuals;
    }


    public float getResidualSumOfSquares()
    {
        return getResiduals().stream().reduce(0.0f, (a, b) -> a + b);
    }


    public float getSlope()
    {
        return slope;
    }


    public float getIntercept()
    {
        return intercept;
    }
}
