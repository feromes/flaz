import matplotlib.pyplot as plt
from flaz import Favela
from pathlib import Path

favela = (
    Favela("São Remo")
    .set_api_path(Path("flaz_tmp"))
    .periodo(2017)
)

res = favela.calc_viela_axis()

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(res["walkable_raw"], cmap="gray")
axs[0, 0].set_title("Walkable (raw)")

axs[0, 1].imshow(res["walkable_dense"], cmap="gray")
axs[0, 1].set_title("After closing")

axs[0, 2].imshow(res["distance"], cmap="inferno")
axs[0, 2].set_title("Distance transform (attribute)")

axs[1, 0].imshow(res["axis"], cmap="hot")
axs[1, 0].set_title("Skeleton (viela axis)")

# Painéis vazios (reservados para próximos passos)
axs[1, 1].axis("off")
axs[1, 2].axis("off")

for ax in axs.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()
