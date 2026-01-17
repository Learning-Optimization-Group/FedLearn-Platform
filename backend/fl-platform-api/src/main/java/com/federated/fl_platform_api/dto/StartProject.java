package com.federated.fl_platform_api.dto;

public class StartProject {

    private String strategy;
    private Integer numRounds;

    public String getStrategy() {
        return strategy;
    }

    public void setStrategy(String strategy) {
        this.strategy = strategy;

    }

    public Integer getNumRounds() {
        return numRounds;
    }

    public void setNumRounds(Integer numRounds) {
        this.numRounds = numRounds;
    }

    @Override
    public String toString() {
        return "StartProject{" +
                "strategy='" + strategy + '\'' +
                ", numRounds=" + numRounds +
                '}';
    }
}
