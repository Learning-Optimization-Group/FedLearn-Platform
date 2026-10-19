package com.federated.fl_platform_api.contract;

import com.fedlearn.contract.v1.ContractIssueCode;

/** One reason a reader refuses an execution contract: an issue code and a ProtoJSON field path. */
public record ContractIssue(ContractIssueCode code, String path) {
}
