-- V37: Fine-grained per-round breakdown timestamps across client and server.
ALTER TABLE round_result ADD COLUMN time_data_loading_s DOUBLE PRECISION;
ALTER TABLE round_result ADD COLUMN time_forward_backward_s DOUBLE PRECISION;
ALTER TABLE round_result ADD COLUMN time_optimizer_step_s DOUBLE PRECISION;
ALTER TABLE round_result ADD COLUMN time_wire_encode_s DOUBLE PRECISION;
ALTER TABLE round_result ADD COLUMN time_aggregation_s DOUBLE PRECISION;
ALTER TABLE round_result ADD COLUMN time_wire_decode_s DOUBLE PRECISION;
ALTER TABLE round_result ADD COLUMN time_eval_s DOUBLE PRECISION;
