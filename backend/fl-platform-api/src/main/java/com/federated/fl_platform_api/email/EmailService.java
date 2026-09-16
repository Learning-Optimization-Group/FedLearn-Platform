package com.federated.fl_platform_api.email;

public interface EmailService {
    /** Send a pre-rendered message. Throws {@link EmailDeliveryException} on transport failure. */
    void send(EmailMessage msg);
}
