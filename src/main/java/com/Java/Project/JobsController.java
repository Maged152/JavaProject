package com.Java.Project;


import org.apache.spark.sql.Row;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import java.util.Iterator;


import static org.apache.spark.sql.functions.col;

@RestController
@RequestMapping(path="/")
public class JobsController {
    private final JobSparkServices jss;

    @Autowired
    public JobsController(JobSparkServices jss) {
        this.jss = jss;
    }

    @GetMapping("/")
    public String index() {
        StringBuilder builder = new StringBuilder();
        builder.append("<h1>Welcome To Wuzzuf Dataset Api:</h1>");

        builder.append("<h2 style=\"text-align:left\">By : Aya , Abdelrahman , Maged and Osama</h2>");

        builder.append("<br><a href=http://localhost:8080/showData>Show data</a> ");

        builder.append("<br><a href=http://localhost:8080/summary>summary of the data</a> " );

        builder.append("<br><a href=http://localhost:8080/schemma>structure of the data</a> " );

        builder.append("<br><a href=http://localhost:8080/dropNulls>drop Nulls</a>  ");

        builder.append("<br><a href=http://localhost:8080/dropDuplicates>Remove Duplicate Rows</a>  ");

        builder.append("<br><a href=http://localhost:8080/getPopularTitles>the most popular job titles</a> ");

        builder.append("<br><a href=http://localhost:8080/countJobs>the jobs for each company </a>  ");

        builder.append("<br><a href=http://localhost:8080/picountJobs>Pi chart for the jobs for each company </a>  ");

        builder.append("<br><a href=http://localhost:8080/bargetPopularTitles>Draw Most Jobs Titles</a> ");

        builder.append("<br><a href=http://localhost:8080/getPopularArea>Most Popular Areas</a> ");

        builder.append("<br><a href=http://localhost:8080/bargetPopularArea> Draw most Popular Areas</a> ");

        builder.append("<br><a href=http://localhost:8080/Print_Skills> Popular Skills</a> ");

        builder.append("<br><a href=http://localhost:8080/Factorize>Factorize YearsExp</a> ");

        builder.append("<br><a href=http://localhost:8080/Prediction>Predict Company</a> ");



        return builder.toString();
    }


    @GetMapping("/showData")  //1 done
    public Object showData()
    {

        StringBuilder jobs = new StringBuilder();
        jobs.append("<h1>Dataset</h1>");
        jobs.append("<table style=\"width:100%\">");
        jobs.append(" <tr>")
                .append("<th>#</th>")
                .append("<th>Title</th>")
                .append("<th>Company</th>")
                .append("<th>Location</th>")
                .append("<th>Type</th>")
                .append("<th>Level</th>")
                .append("<th>YearsEXP</th>")
                .append("<th>Country</th>")
                .append("<th>Skills</th>")
                .append("</tr>");
        int i = 1;
        for (Iterator<Row> it = jss.showData().toLocalIterator(); it.hasNext(); ) {
            Row row = it.next();
            jobs.append(" <tr>")
                    .append("<th>").append(i).append("</th>")
                    .append("<th>").append(row.getString(0)).append("</th>")
                    .append("<th>").append(row.getString(1)).append("</th>")
                    .append("<th>").append(row.getString(2)).append("</th>")
                    .append("<th>").append(row.getString(3)).append("</th>")
                    .append("<th>").append(row.getString(4)).append("</th>")
                    .append("<th>").append(row.getString(5)).append("</th>")
                    .append("<th>").append(row.getString(6)).append("</th>")
                    .append("<th>").append(row.getString(7)).append("</th>")
                    .append("</tr>");
            i++;
        }
        jobs.append("</table>");
        return jobs.toString();
    }

    @GetMapping("/schemma")  //2.2
    public Object getschmma()
    {return jss.getSchema().toString();}


    @GetMapping("/summary")   //2.1 done
    public Object getsummary()
    {
        StringBuilder builder = new StringBuilder();
        builder.append("<h1>Summary</h1>");
        for(Iterator<String> it = jss.getSummary().toJSON().toLocalIterator(); it.hasNext();){
            String js = it.next();
            System.out.println(js);
            builder.append("<br>").append(js).append("</br>");
        }
        return builder.toString();
        //return jss.getSummary().collectAsList().toString();
    }


    @GetMapping("/dropNulls")  //3.1  done
    public Object dropNulls()
    {
        StringBuilder jobs = new StringBuilder();
        jobs.append("<h1>Drop nulls</h1>");
        jobs.append("<table style=\"width:100%\">");
        jobs.append(" <tr>")
                .append("<th>#</th>")
                .append("<th>Title</th>")
                .append("<th>Company</th>")
                .append("<th>Location</th>")
                .append("<th>Type</th>")
                .append("<th>Level</th>")
                .append("<th>YearsEXP</th>")
                .append("<th>Country</th>")
                .append("<th>Skills</th>")
                .append("</tr>");
        int i = 1;
        for (Iterator<Row> it = jss.dropNulls().toLocalIterator(); it.hasNext(); ) {
            Row row = it.next();
            jobs.append(" <tr>")
                    .append("<th>").append(i).append("</th>")
                    .append("<th>").append(row.getString(0)).append("</th>")
                    .append("<th>").append(row.getString(1)).append("</th>")
                    .append("<th>").append(row.getString(2)).append("</th>")
                    .append("<th>").append(row.getString(3)).append("</th>")
                    .append("<th>").append(row.getString(4)).append("</th>")
                    .append("<th>").append(row.getString(5)).append("</th>")
                    .append("<th>").append(row.getString(6)).append("</th>")
                    .append("<th>").append(row.getString(7)).append("</th>")
                    .append("</tr>");
            i++;
        }
        jobs.append("</table>");
        return jobs.toString();

    }

    @GetMapping("/dropDuplicates")  //3.2 done
    public Object dropDuplicates()
    {
        StringBuilder jobs = new StringBuilder();
        jobs.append("<h1>Drop duplicated</h1>");
        jobs.append("<table style=\"width:100%\">");
        jobs.append(" <tr>")
                .append("<th>#</th>")
                .append("<th>Title</th>")
                .append("<th>Company</th>")
                .append("<th>Location</th>")
                .append("<th>Type</th>")
                .append("<th>Level</th>")
                .append("<th>YearsEXP</th>")
                .append("<th>Country</th>")
                .append("<th>Skills</th>")
                .append("</tr>");
        int i = 1;
        for (Iterator<Row> it = jss.dropDuplicates().toLocalIterator(); it.hasNext(); ) {
            Row row = it.next();
            jobs.append(" <tr>")
                    .append("<th>").append(i).append("</th>")
                    .append("<th>").append(row.getString(0)).append("</th>")
                    .append("<th>").append(row.getString(1)).append("</th>")
                    .append("<th>").append(row.getString(2)).append("</th>")
                    .append("<th>").append(row.getString(3)).append("</th>")
                    .append("<th>").append(row.getString(4)).append("</th>")
                    .append("<th>").append(row.getString(5)).append("</th>")
                    .append("<th>").append(row.getString(6)).append("</th>")
                    .append("<th>").append(row.getString(7)).append("</th>")
                    .append("</tr>");
            i++;
        }
        jobs.append("</table>");
        return jobs.toString();
    }

    @GetMapping("/countJobs") //4 done
    public Object countJobs()
    {
        StringBuilder jobs = new StringBuilder();
        jobs.append("<h1>Job count</h1>");
        jobs.append("<table style=\"width:100%\">");
        jobs.append(" <tr>")
                .append("<th>#</th>")
                .append("<th>Company</th>")
                .append("<th>Jobs_Count</th>")
                .append("</tr>");
        int i = 1;
        for (Iterator<Row> it = jss.countJobs().withColumn("Jobs_Count",col("Jobs_Count").cast("String")).toLocalIterator(); it.hasNext(); ) {
            Row row = it.next();
            jobs.append("<tr>")
                    .append("<th>").append(i).append("</th>")
                    .append("<th>").append(row.getString(0)).append("</th>")
                    .append("<th>").append(row.getString(1)).append("</th>")
                    .append("</tr>");
            i++;
        }
        jobs.append("</table>");
        return jobs.toString();

    }


    @GetMapping("/picountJobs") //5 done
    public Object picountJobs()
    {
        StringBuilder builder = new StringBuilder();
        builder.append("<h1>Most Demanded Jobs Pie Chart</h1>");
        String imageByte = DrawChart.drawPieChart(jss.countJobs(), "Most Demanded Jobs");
        builder.append("<img src=\"data:image/jpg;base64, ").append(imageByte).append("\">");
        return builder.toString();

    }

    @GetMapping("/getPopularTitles") //6 done
    public Object getPopularTitles()
    {
        StringBuilder jobs = new StringBuilder();
        jobs.append("<h1>the most popular job titles</h1>");
        jobs.append("<table style=\"width:100%\">");
        jobs.append(" <tr>")
                .append("<th>Title</th>")
                .append("<th>Title_Count</th>")
                .append("</tr>");
        int i = 1;
        for (Iterator<Row> it = jss.getPopularTitles(). withColumn("Title_Count",col("Title_Count").cast("String")).toLocalIterator(); it.hasNext(); ) {
            Row row = it.next();
            jobs.append(" <tr>")
                    .append("<th>").append(i).append("</th>")
                    .append("<th>").append(row.getString(0)).append("</th>")
                    .append("<th>").append(row.getString(1)).append("</th>")
                    .append("</tr>");
            i++;
        }
        jobs.append("</table>");
        return jobs.toString();

    }


    @GetMapping("/bargetPopularTitles") //7 done
    public Object bargetPopularTitles()
    {
        StringBuilder builder = new StringBuilder();
        builder.append("<h1>Most Popular Job Titles Bar Chart</h1>");
        String imageByte = DrawChart.drawBarChart(jss.getPopularTitles(), "Most Popular Jobs titles");
        builder.append("<img src=\"data:image/jpg;base64, ").append(imageByte).append("\">");
        return builder.toString();

    }

    @GetMapping("/getPopularArea")  //8 done
    public Object getPopularArea()
    {
        StringBuilder jobs = new StringBuilder();
        jobs.append("<h1>the most Location</h1>");
        jobs.append("<table style=\"width:100%\">");
        jobs.append(" <tr>")
                .append("<th>Location</th>")
                .append("<th>Location_Count</th>")
                .append("</tr>");
        int i = 1;
        for (Iterator<Row> it = jss.getPopularArea().withColumn("Location_Count",col("Location_Count").cast("String")).toLocalIterator(); it.hasNext(); ) {
            Row row = it.next();
            jobs.append(" <tr>")
                    .append("<th>").append(i).append("</th>")
                    .append("<th>").append(row.getString(0)).append("</th>")
                    .append("<th>").append(row.getString(1)).append("</th>")
                    .append("</tr>");
            i++;
        }
        jobs.append("</table>");
        return jobs.toString();

    }

    @GetMapping("/bargetPopularArea")  //9 done
    public Object bargetPopularArea()
    {
        StringBuilder builder = new StringBuilder();
        builder.append("<h1>Most Popular Job Areas Bar Chart</h1>");
        String imageByte = DrawChart.drawBarChart(jss.getPopularArea(), "Most Popular Jobs titles");
        builder.append("<img src=\"data:image/jpg;base64, ").append(imageByte).append("\">");
        return builder.toString();

    }

    @GetMapping("/Print_Skills")  //10 done
    public Object Print_Skills()
    {
        StringBuilder jobs = new StringBuilder();
        jobs.append("<h1>the most Skills</h1>");
        jobs.append("<table style=\"width:100%\">");
        jobs.append(" <tr>")
                .append("<th>Skills</th>")
                .append("<th>Skills_Rep</th>")
                .append("</tr>");
        int i = 1;
        for (Iterator<Row> it = jss.Print_Skills() .withColumn("Skills_Rep",col("Skills_Rep").cast("String")).toLocalIterator(); it.hasNext(); ) {
            Row row = it.next();
            jobs.append(" <tr>")
                    .append("<th>").append(i).append("</th>")
                    .append("<th>").append(row.getString(0)).append("</th>")
                    .append("<th>").append(row.getString(1)).append("</th>")
                    .append("</tr>");
            i++;
        }
        jobs.append("</table>");
        return jobs.toString();

    }

    @GetMapping("/Factorize") //11 done
    public Object Factorize()
    {
        StringBuilder jobs = new StringBuilder();
        jobs.append("<h1>factorize</h1>");
        jobs.append("<table style=\"width:100%\">");
        jobs.append(" <tr>")
                .append("<th>#</th>")
                .append("<th>Title</th>")
                .append("<th>Company</th>")
                .append("<th>Location</th>")
                .append("<th>Type</th>")
                .append("<th>Level</th>")
                .append("<th>YearsEXP</th>")
                .append("<th>Country</th>")
                .append("<th>Skills</th>")
                .append("<th>Year_EXP_AS_Number</th>")
                .append("</tr>");
        int i = 1;
        for (Iterator<Row> it = jss.Factorize().toLocalIterator(); it.hasNext(); ) {
            Row row = it.next();
            jobs.append(" <tr>")
                    .append("<th>").append(i).append("</th>")
                    .append("<th>").append(row.getString(0)).append("</th>")
                    .append("<th>").append(row.getString(1)).append("</th>")
                    .append("<th>").append(row.getString(2)).append("</th>")
                    .append("<th>").append(row.getString(3)).append("</th>")
                    .append("<th>").append(row.getString(4)).append("</th>")
                    .append("<th>").append(row.getString(5)).append("</th>")
                    .append("<th>").append(row.getString(6)).append("</th>")
                    .append("<th>").append(row.getString(7)).append("</th>")
                    .append("<th>").append(row.getString(8)).append("</th>")
                    .append("</tr>");
            i++;
        }
        jobs.append("</table>");
        return jobs.toString();


    }


    @GetMapping("/Prediction") //12 done
    public Object Predict()
    {
        StringBuilder jobs = new StringBuilder();
        jobs.append("<h1>predictions</h1>");
        jobs.append("<table style=\"width:100%\">");
        jobs.append(" <tr>")
                .append("<th>Title</th>")
                .append("<th>Company</th>")
                .append("<th>Prediction</th>")
                .append("</tr>");
        int i = 1;
        for (Iterator<Row> it = jss.Knearest() .withColumn("Prediction",col("Prediction").cast("String")).toLocalIterator(); it.hasNext(); ) {
            Row row = it.next();
            jobs.append(" <tr>")
                    .append("<th>").append(row.getString(0)).append("</th>")
                    .append("<th>").append(row.getString(1)).append("</th>")
                    .append("<th>").append(row.getString(2)).append("</th>")
                    .append("</tr>");
            i++;
        }
        jobs.append("</table>");
        return jobs.toString();
        //return jss.Knearest();
    }



}
