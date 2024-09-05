package io.github.orionlibs.orion_ml4j.regression;

import io.github.orionlibs.orion_assert.Assert;
import io.github.orionlibs.orion_runnable.functions.OrionFunction1x1;
import io.github.orionlibs.orion_simple_math.statistics.ArithmeticMean;
import io.github.orionlibs.orion_simple_math.statistics.Variance;
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
    private float xMean;
    private float yMean;
    private float numerator;
    private float denominator;
    private List<Float> residuals;
    private float modelError;


    public LinearRegression(List<Pair<Float, Float>> pairsOfInputsOutputsTrainingData, float modelError)
    {
        Assert.notEmpty(pairsOfInputsOutputsTrainingData, "input cannot be empty");
        this.modelError = modelError;
        this.pairsOfInputsOutputsTrainingData = pairsOfInputsOutputsTrainingData;
        this.inputs = pairsOfInputsOutputsTrainingData.stream()
                        .map(pair -> pair.getFirst())
                        .toList();
        this.outputs = pairsOfInputsOutputsTrainingData.stream()
                        .map(pair -> pair.getSecond())
                        .toList();
        this.numerator = 0.0f;
        this.denominator = 0.0f;
    }


    private void runIfItHasNotRunYet()
    {
        if(!hasRegressionRun)
        {
            train();
        }
    }


    public void train()
    {
        xMean = new ArithmeticMean().getMean(inputs);
        yMean = new ArithmeticMean().getMean(outputs);
        numerator = 0.0f;
        denominator = 0.0f;
        for(int i = 0; i < pairsOfInputsOutputsTrainingData.size(); i++)
        {
            float xMinusMean = (inputs.get(i) - xMean);
            numerator += xMinusMean * (outputs.get(i) - yMean);
            denominator += xMinusMean * xMinusMean;
        }
        slope = numerator / denominator;
        intercept = yMean - (slope * xMean);
        modelFunction = (Float x) -> intercept + (slope * x) + modelError;
        residuals = new ArrayList<>();
        for(int i = 0; i < inputs.size(); i++)
        {
            residuals.add(outputs.get(i) - modelFunction.run(inputs.get(i)));
        }
        hasRegressionRun = true;
    }


    public float run(float x)
    {
        runIfItHasNotRunYet();
        return modelFunction.run(x);
    }


    public List<Float> getResiduals()
    {
        runIfItHasNotRunYet();
        return residuals;
    }


    public float getResidualSumOfSquares()
    {
        return getResiduals().stream().reduce(0.0f, (a, b) -> a + b);
    }


    public float getStandardErrorOfOutputsMean()
    {
        runIfItHasNotRunYet();
        return (float)Math.sqrt(Variance.getPopulationVariance(outputs));
    }


    public float getStandardErrorOfIntercept()
    {
        runIfItHasNotRunYet();
        float sum = 0.0f;
        for(int i = 0; i < inputs.size(); i++)
        {
            float xMinusMean = (inputs.get(i) - xMean);
            sum += xMinusMean * xMinusMean;
        }
        float variance = (float)Math.sqrt(Variance.getPopulationVariance(residuals));
        return (float)Math.sqrt(variance * (sum + (xMean * xMean)) / (inputs.size() * sum));
    }


    public float getStandardErrorOfSlope()
    {
        runIfItHasNotRunYet();
        float sum = 0.0f;
        for(int i = 0; i < inputs.size(); i++)
        {
            float xMinusMean = (inputs.get(i) - xMean);
            sum += xMinusMean * xMinusMean;
        }
        float variance = (float)Math.sqrt(Variance.getPopulationVariance(residuals));
        return (float)Math.sqrt(variance / sum);
    }


    public float getResidualStandardError()
    {
        return (float)Math.sqrt(getResidualSumOfSquares() / (inputs.size() - 2));
    }


    public float getTStatistic()
    {
        runIfItHasNotRunYet();
        return slope / getStandardErrorOfSlope();
    }


    public float getTotalSumOfSquares()
    {
        runIfItHasNotRunYet();
        float sum = 0.0f;
        for(int i = 0; i < inputs.size(); i++)
        {
            float yMinusMean = (outputs.get(i) - yMean);
            sum += yMinusMean * yMinusMean;
        }
        return sum;
    }


    public float getRSquaredStatistic()
    {
        float tss = getTotalSumOfSquares();
        return 1 - (getResidualSumOfSquares() / tss);
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
