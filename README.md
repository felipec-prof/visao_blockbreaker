# Block breaker com visão computacional
### Como funciona:

Em um jogo de Block breaker, o objetivo é quebrar blocos usando uma bolinha que ricocheteia ao bater nos cantos da tela ou dentro da forma criada pelo jogador.

Usando visão computacional, são criados 4 pontos na tela - nas pontas dos dedos indicador e polegar de ambas as mãos. Usando esses 4 pontos, é criado um quadrado vermelho que pode fazer a bolinha ricochetear.


<img width="560" height="350" alt="BreakoutHands2026-06-2515-08-35-ezgif com-resize" src="https://github.com/user-attachments/assets/3167eb9f-9f71-435b-b777-73063b4102ec" />

### Como rodar:
- Baixe os dois arquivos do repositório, e os coloque na mesma pasta.
- Caso esteja usando mais de uma webcam, você pode alterar qual quer usar na variável "camera"
  - camera = cv2.VideoCapture(0) - Mude o "0" para "1" caso queira usar uma outra camera conectada ao computador.
- Ao rodar o programa, o jogo já inicia.
- Caso perca, é possível reinicar o jogo com a tecla "R".
- É possível fechar o jogo com a tecla "esc".

### Observações:

- O mediapipe, a biblioteca que faz o reconhecimento de mãos, funciona melhor quando:
  - As mãos estão completamente na tela (sem saírem para fora da borda);
  - As mãos estão mais abertas;
  - Para esse jogo, são apenas usadas duas mãos, então mais de uma mão aparecendo na tela pode atrapahar o jogo;
  - O local tem uma iluminação adequada (nem muito claro ou muito escuro).
- Resultados podem variar dependendo da qualidade da câmera usada.

### Requisitos:
- OpenCV
  - pip install opencv-python
- Mediapipe
  - pip install -q mediapipe
