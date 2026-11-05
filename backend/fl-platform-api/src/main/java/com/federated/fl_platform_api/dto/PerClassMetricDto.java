package com.federated.fl_platform_api.dto;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

/** Per-class precision/recall/F1/support — the "micro" granularity of a classification benchmark. */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class PerClassMetricDto {
    private String label;
    private Double precision;
    private Double recall;
    private Double f1;
    @JsonDeserialize(using = IntegralNumberDeserializers.IntegerValue.class)
    private Integer support;
}
