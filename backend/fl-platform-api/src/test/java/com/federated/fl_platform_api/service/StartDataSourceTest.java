package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.ModelRecipeDto;
import com.federated.fl_platform_api.dto.StartProject;
import com.federated.fl_platform_api.model.TrainingDataSource;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** Which data source a start may ask for: the recipe must offer it, and the run must be one its trainers can join. */
class StartDataSourceTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static ModelRecipeDto recipe(String json) {
        try {
            return MAPPER.readValue(json, ModelRecipeDto.class);
        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            throw new IllegalStateException(e);
        }
    }

    private static final String TINYNET = """
            {"key":"TINYNET_GOLDEN","supported_data_sources":["FIXTURE","LOCAL_SNAPSHOT"]}""";

    private static StartProject start(String dataSource) {
        StartProject s = new StartProject();
        s.setDataSource(dataSource);
        return s;
    }

    @Test
    void theCatalogDtoKeepsTheDataSourcesARecipeSupports() throws Exception {
        assertThat(recipe(TINYNET).supportedDataSources()).containsExactly("FIXTURE", "LOCAL_SNAPSHOT");
        assertThat(recipe("{\"key\":\"CNN\"}").supportedDataSources()).isNull();
    }

    @Test
    void aStartThatNamesNoDataSourceTrainsTheFixture() throws Exception {
        assertThat(ProjectService.resolveDataSource(null, () -> Optional.of(recipe("{\"key\":\"CNN\"}")), "CNN"))
                .isEqualTo(TrainingDataSource.FIXTURE);
        assertThat(ProjectService.resolveDataSource(start(null), () -> Optional.empty(), "CNN"))
                .isEqualTo(TrainingDataSource.FIXTURE);
        assertThat(ProjectService.resolveDataSource(start("FIXTURE"), () -> Optional.empty(), "CNN"))
                .isEqualTo(TrainingDataSource.FIXTURE);
    }

    @Test
    void aRecipeThatOffersParticipantsOwnDataMayTrainOnIt() throws Exception {
        assertThat(ProjectService.resolveDataSource(start("LOCAL_SNAPSHOT"), () -> Optional.of(recipe(TINYNET)),
                "TINYNET_GOLDEN")).isEqualTo(TrainingDataSource.LOCAL_SNAPSHOT);
    }

    @Test
    void aRecipeThatDoesNotOfferParticipantsOwnDataIsRefused() throws Exception {
        for (Optional<ModelRecipeDto> r : List.of(Optional.<ModelRecipeDto>empty(),
                Optional.of(recipe("{\"key\":\"CNN\"}")),
                Optional.of(recipe("{\"key\":\"CNN\",\"supported_data_sources\":[\"FIXTURE\"]}")))) {
            assertThatThrownBy(() -> ProjectService.resolveDataSource(start("LOCAL_SNAPSHOT"), () -> r, "CNN"))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("CNN").hasMessageContaining("own data");
        }
    }

    // Only phones train on their own snapshots, and phones cannot join a secure-aggregation run yet: the run could
    // never complete a round.
    @Test
    void aSecureRunOnParticipantsOwnDataIsRefused() throws Exception {
        StartProject s = start("LOCAL_SNAPSHOT");
        s.setSecureAggregation(true);
        assertThatThrownBy(() -> ProjectService.resolveDataSource(s, () -> Optional.of(recipe(TINYNET)), "TINYNET_GOLDEN"))
                .isInstanceOf(IllegalArgumentException.class).hasMessageContaining("ecure aggregation");
    }
}
