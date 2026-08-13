package com.federated.fl_platform_api.arm;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.federated.fl_platform_api.dto.CreateProjectRequest;
import com.federated.fl_platform_api.dto.ClientConnectionDto;
import com.federated.fl_platform_api.dto.ModelRecipeDto;
import com.federated.fl_platform_api.dto.StartProject;
import com.federated.fl_platform_api.model.TrainingArm;
import jakarta.validation.Validation;
import jakarta.validation.Validator;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * P1-4: the arm reaches the picker with the measured trade-off attached, and the picker's choice
 * reaches the project.
 *
 * <p>Two gaps this closes, both silent:
 *
 * <ol>
 *   <li>{@link ModelRecipeDto} is annotated {@code @JsonIgnoreProperties(ignoreUnknown = true)},
 *       so {@code supported_arms} — which {@code recipes.py --describe} has emitted since P1-1 —
 *       was <em>parsed and dropped</em>. The catalog knew about arms; the frontend never saw them.
 *   <li>{@link CreateProjectRequest} had no arm at all, so a picker had nowhere to send the
 *       choice. P1-2 added the field to {@link StartProject} instead, where no main-source code
 *       reads it: {@code FlServerManager} resolves the arm from {@code project.getTrainingArm()}.
 *       A client could POST an arm to {@code /start}, have it validated, and be silently ignored.
 * </ol>
 */
class ArmCatalogAndCreationTest {

    private static final Validator VALIDATOR =
            Validation.buildDefaultValidatorFactory().getValidator();
    private static final ObjectMapper MAPPER = new ObjectMapper();

    // ---------------------------------------------------------------- catalog carries the arms

    @Test
    void theCatalogDtoKeepsSupportedArms() throws Exception {
        String describeOutput = """
                {"key":"PNEUMONIA_CNN","display_name":"Pneumonia CNN","input_kind":"image",
                 "classes":["NORMAL","PNEUMONIA"],"base_models":["cnn"],"optimizers":["Adam"],
                 "supported_arms":["FULL","FROZEN_HEAD"]}
                """;
        ModelRecipeDto dto = MAPPER.readValue(describeOutput, ModelRecipeDto.class);
        assertThat(dto.supportedArms())
                .as("recipes.py emits supported_arms; the DTO must not silently drop it")
                .containsExactly("FULL", "FROZEN_HEAD");
    }

    @Test
    void theCatalogDtoKeepsTheMeasuredTradeoff() throws Exception {
        String describeOutput = """
                {"key":"PNEUMONIA_CNN","supported_arms":["FULL","FROZEN_HEAD"],
                 "arm_tradeoff":{"headline":"Full fine-tuning buys +0.0224 AUC for 3,321x the communication.",
                                 "comm_ratio":3321,
                                 "caveats":["One task, one alpha, three seeds."]}}
                """;
        ModelRecipeDto dto = MAPPER.readValue(describeOutput, ModelRecipeDto.class);
        assertThat(dto.armTradeoff()).as("the trade-off must survive to the picker").isNotNull();
        assertThat(dto.armTradeoff().headline()).contains("0.0224");
        assertThat(dto.armTradeoff().caveats())
                .as("a number without its caveat is a claim the record does not make")
                .isNotEmpty();
    }

    @Test
    void aRecipeWithoutArmsStillParses() throws Exception {
        // Non-catalog and single-arm recipes omit both fields; the DTO must tolerate that rather
        // than fail the whole catalog load.
        ModelRecipeDto dto = MAPPER.readValue("{\"key\":\"MLP\"}", ModelRecipeDto.class);
        assertThat(dto.supportedArms()).isNull();
        assertThat(dto.armTradeoff()).isNull();
    }

    // ------------------------------------------------------- the picker's choice reaches create

    @Test
    void createAcceptsAValidArm() {
        CreateProjectRequest r = validCreate();
        r.setTrainingArm("FROZEN_HEAD");
        assertThat(VALIDATOR.validate(r)).isEmpty();
    }

    @Test
    void createRejectsAnUnknownArm() {
        CreateProjectRequest r = validCreate();
        r.setTrainingArm("SEMI_FROZEN");
        assertThat(VALIDATOR.validate(r))
                .as("an unrecognised arm must be rejected at the edge, not at FL-server spawn")
                .isNotEmpty();
    }

    @Test
    void createWithoutAnArmIsValidAndMeansFull() {
        CreateProjectRequest r = validCreate();
        assertThat(VALIDATOR.validate(r)).isEmpty();
        assertThat(r.getTrainingArm())
                .as("omitting the arm must keep pre-P1 behaviour; the entity default supplies FULL")
                .isNull();
    }

    @Test
    void everyEnumConstantIsAcceptedByTheCreatePattern() {
        // Guards the split-brain: adding a TrainingArm constant without widening the DTO pattern
        // yields an arm the platform believes is valid and the API rejects.
        for (TrainingArm arm : TrainingArm.values()) {
            CreateProjectRequest r = validCreate();
            r.setTrainingArm(arm.name());
            assertThat(VALIDATOR.validate(r))
                    .as("CreateProjectRequest rejects TrainingArm." + arm)
                    .isEmpty();
        }
    }

    @Test
    void addingTheArmDidNotDetachTaskTypeValidation() {
        // Adding a field immediately above `private String taskType` landed BETWEEN that field and
        // its @Pattern, silently transferring the SEQ_CLASSIFICATION|CAUSAL_LM constraint onto the
        // new field and leaving taskType unvalidated. The visible half (the arm rejecting valid
        // values) failed loudly; this half would not have. Pin it.
        CreateProjectRequest r = validCreate();
        r.setTaskType("NOT_A_TASK_TYPE");
        assertThat(VALIDATOR.validate(r))
                .as("taskType lost its @Pattern — a constraint was detached from its field")
                .isNotEmpty();
    }

    @Test
    void theProjectResponseAlwaysStatesAnArm() {
        // Never null on the wire: a UI that has to treat "absent" as FULL would be re-deriving a
        // default the server already knows, and a frozen project would be indistinguishable from a
        // full one in every list view.
        com.federated.fl_platform_api.model.Project p = new com.federated.fl_platform_api.model.Project();
        assertThat(p.getTrainingArm())
                .as("the entity default supplies FULL, which is what the response must report")
                .isEqualTo(TrainingArm.FULL);
    }

    // ------------------------------------------------- the arm reaches the client (P1-5)

    @Test
    void theConnectionPayloadCarriesTheArm() {
        // GET /client/projects/{id}/connection is how a client learns its launch config. Before
        // P1-5 it carried the strategy but not the arm, so a FROZEN_HEAD project had the server
        // filtering to the head subset while the client uploaded the full state dict.
        ClientConnectionDto dto = new ClientConnectionDto();
        dto.setTrainingArm(TrainingArm.FROZEN_HEAD.name());
        assertThat(dto.getTrainingArm()).isEqualTo("FROZEN_HEAD");
    }

    @Test
    void theConnectionPayloadStatesFullRatherThanOmittingIt() {
        // The service always sets an arm (entity default FULL). Omission would make the client
        // infer FULL from silence, which is indistinguishable from a backend that predates P1 —
        // and the two want different behaviour.
        ClientConnectionDto dto = new ClientConnectionDto();
        dto.setTrainingArm(TrainingArm.FULL.name());
        assertThat(dto.getTrainingArm()).isEqualTo("FULL");
    }

    private CreateProjectRequest validCreate() {
        CreateProjectRequest r = new CreateProjectRequest();
        r.setName("arm-test");
        r.setModelType("PNEUMONIA_CNN");
        r.setModelName("pneumonia_cnn");
        r.setOptimizer("Adam");
        r.setPretrainEpochs(0);   // @NotNull
        return r;
    }
}
