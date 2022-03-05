package com.Java.Project;


import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.SparkSession;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;
import org.springframework.context.support.PropertySourcesPlaceholderConfigurer;


@Configuration
//@PropertySource("classpath:application.properties")
public class SparkConf {

    @Value("${app.name:spark-boot}")
    private String appName;

    @Value("${spark.home}")
    private String sparkHome;

    @Value("${master.uri:local}")
    private String masterUri;


    @Bean
    public SparkSession sparkSession() {
        SparkSession sp=SparkSession
                .builder()
                .appName("Spark Example")
                .master("local[4]")
                .getOrCreate();

        return sp;
    }

}
