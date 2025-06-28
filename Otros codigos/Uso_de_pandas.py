import pandas as pd

# crear un DataFrame con dos columnas de números aleatorios
df = pd.DataFrame({'columna_1': [1, 2, 3, 4], 'columna_2': [5, 6, 7, 8]})

# agregar una nueva columna calculada como la suma de las otras dos columnas
df['columna_3'] = df['columna_1'] + df['columna_2']

# imprimir el DataFrame
print(df)





""" Por si no es válido el DESNET por no hacer embeddings xd"""
""" 
#OPCION 2: CREAR LA CNN PROPIA

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # Entrada: 1 canal (blanco y negro), Salida: 16 canales
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 256x256 → 256x256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 256x256 → 128x128

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 128x128 → 128x128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 128x128 → 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x64 → 64x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # 64x64 → 32x32
        )

        # Capa totalmente conectada
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),  # Aplana y conecta
            nn.ReLU(),
            nn.Linear(128, 1)  # Salida binaria
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Aplana (flatten)
        x = self.fc_layers(x)
        return x 
"""