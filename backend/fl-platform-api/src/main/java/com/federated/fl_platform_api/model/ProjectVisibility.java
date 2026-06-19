package com.federated.fl_platform_api.model;

/**
 * Project visibility tiers (spec §Effort B):
 * <ul>
 *   <li>{@code PUBLIC}    — discoverable; any org member may join and train freely (auto-join).</li>
 *   <li>{@code RESTRICTED}— discoverable; joining requires an access request the owner approves.</li>
 *   <li>{@code PRIVATE}   — hidden from discovery; the owner adds participants by invitation only.</li>
 * </ul>
 */
public enum ProjectVisibility {
    PUBLIC, RESTRICTED, PRIVATE
}
