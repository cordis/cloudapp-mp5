import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.tree.RandomForest;

import java.util.Arrays;
import java.util.HashMap;
import java.util.regex.Pattern;


public final class RandomForestMP {
    private static Pattern DELIMITER = Pattern.compile(",");

    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println("Usage: RandomForestMP <training_data> <test_data> <results>");
            System.exit(1);
        }
        String training_data_path = args[0];
        String test_data_path = args[1];
        String results_path = args[2];

        SparkConf sparkConf = new SparkConf().setAppName("RandomForestMP");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        JavaRDD<LabeledPoint> trainingLabeledPointList = sc.textFile(training_data_path).map(new ParseLabeledPoint());
        JavaRDD<String> testLineList = sc.textFile(test_data_path);

        Integer numClasses = 2;
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        Integer numTrees = 3;
        String featureSubsetStrategy = "auto";
        String impurity = "gini";
        Integer maxDepth = 5;
        Integer maxBins = 32;
        Integer seed = 12345;

        RandomForestModel model = RandomForest.trainClassifier(
                trainingLabeledPointList,
                numClasses,
                categoricalFeaturesInfo,
                numTrees,
                featureSubsetStrategy,
                impurity,
                maxDepth,
                maxBins,
                seed
        );

        JavaRDD<LabeledPoint> results = testLineList
                .map(new ParseFeatures())
                .map(new Classifier(model));

        results.saveAsTextFile(results_path);

        sc.stop();
    }

    private static class ParseFeatures implements Function<String, Vector> {
        @Override
        public Vector call(String line) throws Exception {
            String[] tokenList = DELIMITER.split(line);
            double[] valueList = parseValueList(tokenList);
            return makeFeatures(valueList);
        }

    }

    private static class ParseLabeledPoint implements Function<String, LabeledPoint> {
        @Override
        public LabeledPoint call(String line) throws Exception {
            String[] tokenList = DELIMITER.split(line);
            double[] valueList = parseValueList(tokenList);
            Vector features = makeFeatures(valueList);
            Double label = valueList[valueList.length - 1];
            return new LabeledPoint(label, features);
        }

    }

    static private double[] parseValueList(String[] tokenList) {
        double[] valueList = new double[tokenList.length - 1];
        for (int i = 0; i < tokenList.length - 1; i++) {
            valueList[i] = Double.parseDouble(tokenList[i]);
        }
        return valueList;
    }

    static private Vector makeFeatures(double[] valueList) {
        return Vectors.dense(Arrays.copyOfRange(valueList, 0, valueList.length - 1));
    }

    private static class Classifier implements Function<Vector, LabeledPoint> {
        final private RandomForestModel model;

        public Classifier(RandomForestModel model) {
            this.model = model;
        }

        @Override
        public LabeledPoint call(Vector features) throws Exception {
            return new LabeledPoint(this.model.predict(features), features);
        }

    }

}
