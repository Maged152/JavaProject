package com.Java.Project;




class Jobs  {
    public String Title;
    public String Company;
    public String Location;
    public String Type;
    public String Level;
    public String YearsExp;
    public String Country;
    public String Skills;

    public Jobs()  {
    }

    public Jobs(String title, String company, String location, String type, String level, String yearsExp, String country, String skills) {
        Title = title;
        Company = company;
        Location = location;
        Type = type;
        Level = level;
        YearsExp = yearsExp;
        Country = country;
        Skills = skills;
    }

    public String getTitle() {
        return Title;
    }

    public String getCompany() {
        return Company;
    }

    public String getLocation() {
        return Location;
    }

    public String getType() {
        return Type;
    }

    public String getLevel() {
        return Level;
    }

    public String getYearsExp() {
        return YearsExp;
    }

    public String getCountry() {
        return Country;
    }

    public String getSkills() {
        return Skills;
    }

    public void setTitle(String title) {
        Title = title;
    }

    public void setCompany(String company) {
        Company = company;
    }

    public void setLocation(String location) {
        Location = location;
    }

    public void setType(String type) {
        Type = type;
    }

    public void setLevel(String level) {
        Level = level;
    }

    public void setYearsExp(String yearsExp) {
        YearsExp = yearsExp;
    }

    public void setCountry(String country) {
        Country = country;
    }

    public void setSkills(String skills) {
        Skills = skills;
    }

      @Override
      public String toString() {
          return "Jobs{" +
                  "Title='" + Title + '\'' +
                  ", Company='" + Company + '\'' +
                  ", Location='" + Location + '\'' +
                  ", Type='" + Type + '\'' +
                  ", Level='" + Level + '\'' +
                  ", YearsExp='" + YearsExp + '\'' +
                  ", Country='" + Country + '\'' +
                  ", Skills='" + Skills + '\'' +
                  '}';
      }
  }
