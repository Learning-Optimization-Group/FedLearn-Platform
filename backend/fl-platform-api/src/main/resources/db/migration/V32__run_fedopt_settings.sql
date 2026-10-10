ALTER TABLE runs ADD COLUMN fedopt_variant varchar(16),
    ADD COLUMN fedopt_server_learning_rate double precision,
    ADD COLUMN fedopt_beta1 double precision,
    ADD COLUMN fedopt_beta2 double precision,
    ADD COLUMN fedopt_tau double precision;

-- Earlier FedOpt servers used these fixed settings.
UPDATE runs SET fedopt_variant = 'adam', fedopt_server_learning_rate = 1.0,
    fedopt_beta1 = 0.9, fedopt_beta2 = 0.99, fedopt_tau = 0.001 WHERE strategy = 'FedOpt';

ALTER TABLE runs ADD CONSTRAINT runs_fedopt_settings CHECK (
    (strategy = 'FedOpt' AND fedopt_variant IS NOT NULL AND fedopt_variant IN ('adam', 'yogi')
        AND fedopt_server_learning_rate IS NOT NULL AND fedopt_server_learning_rate > 0 AND fedopt_server_learning_rate <= 10
        AND fedopt_beta1 IS NOT NULL AND fedopt_beta1 >= 0 AND fedopt_beta1 < 1
        AND fedopt_beta2 IS NOT NULL AND fedopt_beta2 >= 0 AND fedopt_beta2 < 1
        AND fedopt_tau IS NOT NULL AND fedopt_tau > 0 AND fedopt_tau <= 1)
    OR (strategy <> 'FedOpt' AND fedopt_variant IS NULL AND fedopt_server_learning_rate IS NULL
        AND fedopt_beta1 IS NULL AND fedopt_beta2 IS NULL AND fedopt_tau IS NULL));
