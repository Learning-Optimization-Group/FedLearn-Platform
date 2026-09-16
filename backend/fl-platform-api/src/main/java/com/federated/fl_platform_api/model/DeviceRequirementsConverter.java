package com.federated.fl_platform_api.model;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.federated.fl_platform_api.dto.DeviceRequirements;
import jakarta.persistence.AttributeConverter;
import jakarta.persistence.Converter;

@Converter
public class DeviceRequirementsConverter implements AttributeConverter<DeviceRequirements, String> {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Override
    public String convertToDatabaseColumn(DeviceRequirements attribute) {
        if (attribute == null) return null;
        try {
            return MAPPER.writeValueAsString(attribute);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to serialize requirements override", e);
        }
    }

    @Override
    public DeviceRequirements convertToEntityAttribute(String dbData) {
        if (dbData == null || dbData.isBlank()) return null;
        try {
            return MAPPER.readValue(dbData, DeviceRequirements.class);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to deserialize requirements override", e);
        }
    }
}
