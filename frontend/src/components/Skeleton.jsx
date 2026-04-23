/**
 * Skeleton.jsx
 * Shimmer placeholder components that match the shape of each viz section.
 * Pure CSS animation — no external library.
 */

function SkeletonBox({ width = "100%", height = 20, style = {} }) {
    return (
        <div
            className="skeleton"
            style={{ width, height, borderRadius: 6, ...style }}
        />
    );
}

/** Mimics the pills row in TokenView */
export function TokenSkeleton() {
    return (
        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            <SkeletonBox height={16} width="40%" />
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                {Array.from({ length: 8 }).map((_, i) => (
                    <SkeletonBox key={i} width={64} height={48} />
                ))}
            </div>
        </div>
    );
}

/** Mimics the N×N heatmap grid */
export function HeatmapSkeleton() {
    return (
        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            <SkeletonBox height={16} width="50%" />
            <div style={{ display: "flex", gap: "0.35rem" }}>
                {Array.from({ length: 6 }).map((_, i) => (
                    <SkeletonBox key={i} width={36} height={28} />
                ))}
            </div>
            <SkeletonBox height={260} />
        </div>
    );
}

/** Mimics the 384-dim bar charts */
export function EmbedSkeleton() {
    return (
        <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
            {Array.from({ length: 3 }).map((_, i) => (
                <div key={i} style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                    <SkeletonBox height={12} width="30%" />
                    <SkeletonBox height={60} />
                </div>
            ))}
        </div>
    );
}

/** Mimics the pooling animation rows */
export function PoolSkeleton() {
    return (
        <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
                    <SkeletonBox width={72} height={10} />
                    <SkeletonBox height={10} style={{ flex: 1 }} />
                </div>
            ))}
        </div>
    );
}

/** Generic single-block skeleton */
export function GenericSkeleton() {
    return (
        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            <SkeletonBox height={16} width="45%" />
            <SkeletonBox height={120} />
        </div>
    );
}

export default SkeletonBox;
