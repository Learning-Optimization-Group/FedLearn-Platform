package com.federated.fl_platform_api.contract;

/** A fact an execution contract needs is absent or cannot be stated in contract v1; the message says which. */
public class NotRepresentableException extends Exception {

    public NotRepresentableException(String message) {
        super(message);
    }
}
