import pandas as pd
import matplotlib.pyplot as plt

# 1. LOAD DATA
df = pd.read_csv("Iris.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
df.info()
# 2. DATA CLEANING
df.drop("Id", axis=1, inplace=True)
 
df.dropna(inplace=True)

 
df.drop_duplicates(inplace=True)

print("\nNull values:")
print(df.isnull().sum())

print("\nDataset shape:", df.shape)
# 3. SPECIES COUNT
print("\nSpecies:")
print(df["Species"].unique())

print("\nSpecies Counts:")
print(df["Species"].value_counts())
# 4. SPECIES COUNT - BAR CHART
species_counts = df["Species"].value_counts()

plt.figure(figsize=(7, 5))

plt.bar(
    species_counts.index,
    species_counts.values
)

plt.title("Number of Flowers in Each Species")
plt.xlabel("Species")
plt.ylabel("Number of Flowers")

plt.show()
# 5. FEATURE DISTRIBUTION - HISTOGRAM
features = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

df[features].hist(
    figsize=(10, 8),
    bins=10
)

plt.suptitle("Distribution of Iris Features")

plt.show()
# 6. FEATURE DISTRIBUTION - BOXPLOT
plt.figure(figsize=(10, 6))

plt.boxplot(
    [
        df["SepalLengthCm"],
        df["SepalWidthCm"],
        df["PetalLengthCm"],
        df["PetalWidthCm"]
    ],
    tick_labels=[
        "Sepal Length",
        "Sepal Width",
        "Petal Length",
        "Petal Width"
    ]
)

plt.title("Distribution of Iris Features")
plt.xlabel("Features")
plt.ylabel("Measurement (cm)")

plt.show()
# 7. FEATURE RELATIONSHIP - SEPAL
plt.figure(figsize=(8, 6))

for species in df["Species"].unique():

    data = df[df["Species"] == species]

    plt.scatter(
        data["SepalLengthCm"],
        data["SepalWidthCm"],
        label=species
    )

plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend()

plt.show()

# 8. FEATURE RELATIONSHIP - PETAL
plt.figure(figsize=(8, 6))

for species in df["Species"].unique():

    data = df[df["Species"] == species]

    plt.scatter(
        data["PetalLengthCm"],
        data["PetalWidthCm"],
        label=species
    )

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()

plt.show()
# 9. CORRELATION MATRIX
correlation = df[features].corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(8, 6))

plt.imshow(correlation)

plt.colorbar()

plt.xticks(
    range(len(features)),
    features,
    rotation=45
)

plt.yticks(
    range(len(features)),
    features
)

plt.title("Feature Correlation Matrix")

# Display correlation values
for i in range(len(features)):
    for j in range(len(features)):

        plt.text(
            j,
            i,
            round(correlation.iloc[i, j], 2),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.show()