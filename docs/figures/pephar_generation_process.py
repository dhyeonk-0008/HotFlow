"""PepHAR peptide generation process visualization."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(6, 1, figsize=(14, 16), gridspec_kw={"hspace": 0.5})

L = 12  # peptide length for illustration
anchor_positions = [3, 8]  # 0-indexed anchor positions (2 anchors)
K = len(anchor_positions)

# Colors
C_EMPTY = "#E0E0E0"       # not yet generated
C_ANCHOR = "#E74C3C"      # anchor/hotspot residue
C_GENERATED = "#3498DB"   # newly generated residue
C_PREV = "#85C1E9"        # previously generated residue
C_RECEPTOR = "#2ECC71"    # receptor surface

residue_radius = 0.35
x_start = 1.0
x_spacing = 1.0


def draw_receptor(ax, x_center, y=-1.2, width=12):
    """Draw receptor surface as a wavy shape at the bottom."""
    x = np.linspace(x_center - width / 2, x_center + width / 2, 200)
    y_top = y + 0.3 * np.sin(2 * np.pi * x / 3) + 0.15 * np.cos(4 * np.pi * x / 5)
    y_bot = np.full_like(x, y - 1.5)
    ax.fill_between(x, y_bot, y_top, color=C_RECEPTOR, alpha=0.25, zorder=0)
    ax.plot(x, y_top, color=C_RECEPTOR, alpha=0.6, lw=1.5, zorder=0)
    ax.text(x_center, y - 0.7, "Receptor surface", ha="center", va="center",
            fontsize=9, color="#1a8a4a", fontstyle="italic")


def draw_residues(ax, states, highlight=None, arrows=None):
    """
    Draw a row of residues.
    states: list of ('empty', 'anchor', 'generated', 'prev')
    highlight: set of indices to draw with thicker border
    arrows: list of (from_idx, to_idx, label) for extension arrows
    """
    x_center = x_start + (L - 1) * x_spacing / 2
    draw_receptor(ax, x_center)

    color_map = {
        "empty": C_EMPTY,
        "anchor": C_ANCHOR,
        "generated": C_GENERATED,
        "prev": C_PREV,
    }

    for i, state in enumerate(states):
        x = x_start + i * x_spacing
        y = 0
        color = color_map[state]
        lw = 2.5 if (highlight and i in highlight) else 1.0
        ec = "#2C3E50" if (highlight and i in highlight) else "#7F8C8D"

        circle = plt.Circle((x, y), residue_radius, fc=color, ec=ec, lw=lw, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=9,
                fontweight="bold" if state != "empty" else "normal",
                color="white" if state in ("anchor", "generated") else "#999")

        # Draw dashed connection lines for hotspot → receptor
        if state == "anchor":
            ax.plot([x, x], [-residue_radius, -0.85], ls="--", color=C_ANCHOR,
                    alpha=0.4, lw=1.2, zorder=1)

    # Draw backbone connections
    for i in range(L - 1):
        if states[i] != "empty" and states[i + 1] != "empty":
            x1 = x_start + i * x_spacing + residue_radius
            x2 = x_start + (i + 1) * x_spacing - residue_radius
            ax.plot([x1, x2], [0, 0], color="#2C3E50", lw=1.5, zorder=1)

    if arrows:
        for from_idx, to_idx, label in arrows:
            x_from = x_start + from_idx * x_spacing
            x_to = x_start + to_idx * x_spacing
            direction = 1 if to_idx > from_idx else -1
            ax.annotate(
                "", xy=(x_to, 0.65), xytext=(x_from, 0.65),
                arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.15",
                                color="#E67E22", lw=2),
                zorder=3,
            )
            mid_x = (x_from + x_to) / 2
            ax.text(mid_x, 0.95, label, ha="center", va="bottom",
                    fontsize=8, color="#E67E22", fontweight="bold")

    ax.set_xlim(-0.5, x_start + L * x_spacing + 0.5)
    ax.set_ylim(-2.5, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")


# ── Step 0: Initial state ──
states0 = ["empty"] * L
draw_residues(axes[0], states0)
axes[0].set_title("Step 0: Initial state — peptide length L=12, receptor given",
                   fontsize=12, fontweight="bold", pad=10, loc="left")
axes[0].text(x_start + (L - 1) * x_spacing / 2, 1.6,
             "EBM density model scores receptor surface → identify hotspot locations",
             ha="center", fontsize=9, color="#555", fontstyle="italic")

# ── Step 1: Anchor generation ──
states1 = ["empty"] * L
for p in anchor_positions:
    states1[p] = "anchor"
draw_residues(axes[1], states1, highlight=set(anchor_positions))
axes[1].set_title("Step 1: Generate K anchor (hotspot) residues via EBM sampling",
                   fontsize=12, fontweight="bold", pad=10, loc="left")
axes[1].text(x_start + (L - 1) * x_spacing / 2, 1.6,
             "Anchor positions chosen by even spacing  |  EBM optimizes (coords, AA type)",
             ha="center", fontsize=9, color="#555", fontstyle="italic")

# ── Step 2: First expansion round ──
states2 = ["empty"] * L
for p in anchor_positions:
    states2[p] = "prev"
# Expand: anchor[0] backward → pos 2, anchor[0] forward → pos 4
states2[2] = "generated"
states2[4] = "generated"
# Expand: anchor[1] backward → pos 7, anchor[1] forward → pos 9
states2[7] = "generated"
states2[9] = "generated"
draw_residues(axes[2], states2, highlight={2, 4, 7, 9},
              arrows=[(3, 2, "← N-term"), (3, 4, "C-term →"),
                      (8, 7, "← N-term"), (8, 9, "C-term →")])
axes[2].set_title("Step 2: Bidirectional extension — first round (1 residue each direction per anchor)",
                   fontsize=12, fontweight="bold", pad=10, loc="left")

# ── Step 3: Continue expansion ──
states3 = ["empty"] * L
for p in anchor_positions:
    states3[p] = "prev"
for p in [2, 4, 7, 9]:
    states3[p] = "prev"
# Next round
states3[1] = "generated"
states3[5] = "generated"
states3[6] = "generated"
states3[10] = "generated"
draw_residues(axes[3], states3, highlight={1, 5, 6, 10},
              arrows=[(2, 1, "←"), (4, 5, "→"), (7, 6, "←"), (9, 10, "→")])
axes[3].set_title("Step 3: Continue bidirectional extension — round 2",
                   fontsize=12, fontweight="bold", pad=10, loc="left")

# ── Step 4: Final residues ──
states4 = ["prev"] * L
states4[0] = "generated"
states4[11] = "generated"
# merge fragments at pos 5-6 already connected
draw_residues(axes[4], states4, highlight={0, 11},
              arrows=[(1, 0, "←"), (10, 11, "→")])
axes[4].set_title("Step 4: Fill remaining termini — fragments merge into complete chain",
                   fontsize=12, fontweight="bold", pad=10, loc="left")

# ── Step 5: Complete peptide + optional fine-tuning ──
states5 = ["prev"] * L
for p in anchor_positions:
    states5[p] = "anchor"
draw_residues(axes[5], states5)
axes[5].set_title("Step 5: Complete peptide — optional joint fine-tuning of all residues",
                   fontsize=12, fontweight="bold", pad=10, loc="left")
axes[5].text(x_start + (L - 1) * x_spacing / 2, 1.6,
             "Fine-tune: jointly optimize all coords + dihedral angles + AA identities",
             ha="center", fontsize=9, color="#555", fontstyle="italic")

# ── Legend ──
legend_patches = [
    mpatches.Patch(fc=C_ANCHOR, ec="#2C3E50", label="Anchor (hotspot) residue"),
    mpatches.Patch(fc=C_GENERATED, ec="#2C3E50", label="Newly generated residue"),
    mpatches.Patch(fc=C_PREV, ec="#7F8C8D", label="Previously generated residue"),
    mpatches.Patch(fc=C_EMPTY, ec="#7F8C8D", label="Not yet generated"),
    mpatches.Patch(fc=C_RECEPTOR, ec=C_RECEPTOR, alpha=0.3, label="Receptor surface"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=5, fontsize=10,
           frameon=True, fancybox=True, shadow=False,
           bbox_to_anchor=(0.5, -0.01))

fig.suptitle("PepHAR: Anchor-Based Autoregressive Peptide Generation",
             fontsize=16, fontweight="bold", y=0.995)

plt.savefig("figures/pephar_generation_process.png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.savefig("figures/pephar_generation_process.pdf", bbox_inches="tight",
            facecolor="white")
print("Saved to figures/pephar_generation_process.png/.pdf")
