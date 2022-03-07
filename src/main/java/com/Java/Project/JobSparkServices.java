package com.Java.Project;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.StructType;
import org.knowm.xchart.*;
import org.knowm.xchart.style.Styler;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.apache.spark.sql.functions.*;

@Service
public class JobSparkServices {

    private SparkSession sp;
    private Dataset<Row> ds;

    @Autowired
    public JobSparkServices(SparkSession sp) {
        this.sp = sp;
        this.ds = this.sp.read().option("header", true).csv("src/main/resources/Wuzzuf_Jobs.csv");

    }
    public Object getTable(){

        String item="";
        for(Iterator<Row> iter = ds.toLocalIterator(); iter.hasNext();) {
            item+= (iter.next()).toString();
        }
        return  item;
    }
    //******************************///
    // 1. Read the dataset, convert it to DataFrame or Spark Dataset, and display some from it.
    public Dataset<Row> showData(){
        return ds.limit(50);
    }
    //2. Display structure and summary of the data.
    public StructType getSchema(){
        return ds.schema();
    }
    public Dataset<Row> getSummary(){
        return ds.describe();
    }
    //3. Clean the data (null, duplications)
    public Dataset<Row> dropNulls(){
        return ds.filter(row -> !row.anyNull());
        //return getDataset().na().drop();
    }
    public Dataset<Row> dropDuplicates(){
        return ds.distinct();
    }
    //4. Count the jobs for each company and display that in order (What are the most demanding companies for jobs?)
    public Dataset<Row> countJobs(){
        ds.createOrReplaceTempView("Jobs");
        //return ds.groupBy("Company").count().orderBy(col("count").desc());
        return sp.sql("SELECT Company, count(Title) as Jobs_Count FROM Jobs group by Company order by Jobs_Count desc");
    }
    //6. Find out what are the most popular job titles.
    public Dataset<Row> getPopularTitles(){
        ds.createOrReplaceTempView("titles");
        return sp.sql("SELECT Title, count(Title) as Title_Count from titles group by Title order by Title_Count desc" );
    }
    //8. Find out the most popular areas?
    public Dataset<Row> getPopularArea(){
        ds.createOrReplaceTempView("Areas");
        return sp.sql("SELECT Location, count(Location) as Location_Count FROM Areas group by Location order by Location_Count desc");
    }
    //10.Print skills one by one and how many each repeated and order the output to find out the most important skills required?
    public Dataset<Row> getPopularSkills(){

        ds.createOrReplaceTempView("Skills_Job");
        Dataset<Row> dd=sp.sql("SELECT Skills from Skills_Job");
        dd= dd.withColumn("Skills",explode(split(col("Skills"),",")));
        dd.createOrReplaceTempView("Skill_Job");

        return sp.sql("SELECT Skills, count(Skills) as Skills_Rep FROM Skill_Job group by Skills order by Skills_Rep desc");

    }
    //11- Factorize exp
    public Dataset<Row> Factorize(){

        Dataset<Row>ds2=ds.withColumn("Year_EXP_AS_Number", functions.regexp_replace(ds.col("YearsExp"), "[a-zA-Z]", ""));
        StringIndexer indexer = new StringIndexer()
                .setInputCol("Year_EXP_AS_Number")
                .setOutputCol("Year_EXP_AS_Code");

        return indexer.fit(ds2).transform(ds2).withColumn("Year_EXP_AS_Code",col("Year_EXP_AS_Code").cast("Integer"))
                .withColumn("Year_EXP_AS_Code",col("Year_EXP_AS_Code").cast("String")).drop("Year_EXP_AS_Number");


    }
    //12 k-nearest n
    public Dataset<Row> Knearest(){

            //Spliting Data
            Dataset<Row> s = ds.select("Title","Company");
            Dataset<Row> dsarr[] = s.randomSplit (new double[]{0.8, 0.2}, 42);

            //Preprocessing
            StringIndexer indexer_title= new StringIndexer()
                    .setInputCol("Title")
                    .setOutputCol("title_Index");

            StringIndexer indexer_company = new StringIndexer()
                    .setInputCol("Company")
                    .setOutputCol("company_Index");

            Dataset<Row> indexed_train_title= indexer_title.fit(dsarr[0]).transform(dsarr[0]);
            Dataset<Row> indexed_train_final = indexer_company.fit(indexed_train_title).transform(indexed_train_title);

            Dataset<Row> indexed_test_title = indexer_title.fit(dsarr[1]).transform(dsarr[1]);
            Dataset<Row> indexed_test_final = indexer_company.fit(indexed_test_title).transform(indexed_test_title);


            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(new String[]{"title_Index", "company_Index"})
                    .setOutputCol("features");

            Dataset<Row> train = assembler.transform(indexed_train_final.na().drop());
            Dataset<Row> test = assembler.transform(indexed_test_final.na().drop());

            //Training
            KMeans kmeans = new KMeans().setK(5).setSeed(1L);
            kmeans.setFeaturesCol("features");

            //Testing
            KMeansModel model = kmeans.fit(train);
            // Making predictions
            Dataset<Row> predictions = model.transform(test);
            //predictions.drop("title_Index","company_index","features");

            return predictions.drop("title_Index","company_index","features");
        }
    //*****************************************//
    }




