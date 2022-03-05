package com.Java.Project;



import org.apache.commons.net.util.Base64;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.knowm.xchart.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DrawChart {

    public static String drawPieChart(Dataset<Row> dataset, String title){
        PieChart chart = new PieChartBuilder().width(1000).height(600).title(title).build();
        List<Row> rows = dataset.select(dataset.columns()[0],dataset.columns()[1]).limit(10).collectAsList();
        for (Row row : rows){
            chart.addSeries(row.getString(0), row.getLong(1));
        }
        byte[] imageBytes = new byte[]{};
        try {
            imageBytes = BitmapEncoder.getBitmapBytes(chart, BitmapEncoder.BitmapFormat.JPG);
        } catch (IOException e) {
            e.printStackTrace(); // Or something more intellegent
        }
        return Base64.encodeBase64String(imageBytes);
    }

    public static String drawBarChart(Dataset<Row> dataset, String title){

        CategoryChart chart = new CategoryChartBuilder().width(1000).height(600).title(title)
                .xAxisTitle(dataset.columns()[0])
                .yAxisTitle(dataset.columns()[1]).build();
        List<Row> rows = dataset.select(dataset.columns()[0],dataset.columns()[1]).limit(5).collectAsList();
        List<String> keys = new ArrayList<>();
        List<Long> values = new ArrayList<>();
        for (Row row : rows){
            keys.add(row.getString(0));
            values.add(row.getLong(1));
        }
        chart.getStyler().setHasAnnotations(true);
        chart.addSeries(dataset.columns()[0],keys,values);
        byte[] imageBytes = new byte[]{};
        try {
            imageBytes = BitmapEncoder.getBitmapBytes(chart, BitmapEncoder.BitmapFormat.JPG);
        } catch (IOException e) {
            e.printStackTrace(); // Or something more intellegent
        }
        return Base64.encodeBase64String(imageBytes);
    }

}
