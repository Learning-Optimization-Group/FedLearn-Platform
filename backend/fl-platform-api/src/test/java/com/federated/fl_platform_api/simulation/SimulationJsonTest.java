package com.federated.fl_platform_api.simulation;

import org.junit.jupiter.api.Test;
import java.io.InputStream;
import static org.assertj.core.api.Assertions.*;

class SimulationJsonTest {
    @Test void oversizedStreamingBodyIsRejectedWithoutReadingTheRest() {
        class EndlessBody extends InputStream {
            int consumed;
            @Override public int read() { consumed++; return 'a'; }
        }
        var body = new EndlessBody();
        assertThatThrownBy(() -> SimulationJson.readChoices(body)).isInstanceOf(SimulationException.class);
        assertThat(body.consumed).isLessThanOrEqualTo(65537);
    }
}
