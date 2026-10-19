package com.federated.fl_platform_api.contract;

/** The input does not parse as an execution contract. */
public class MalformedContractException extends Exception {

    public MalformedContractException(String message, Throwable cause) {
        super(message, cause);
    }
}
