import socket

def enviar_comando_monitor(mensagem_str, host="127.0.0.1", port=60001):
    """
    Abre conexão com o Monitor, envia a string do comando e fecha.
    O protocolo do rcssservermj exige que o tamanho da mensagem seja
    enviado nos primeiros 4 bytes (big endian).
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        
        msg_bytes = mensagem_str.encode('utf-8')
        # Empacota os 4 bytes do tamanho antes da mensagem
        sock.send(len(msg_bytes).to_bytes(4, byteorder='big') + msg_bytes)
        sock.close()
    except ConnectionRefusedError:
        print("Erro: Não foi possível conectar. O rcssservermj está rodando e a porta 60001 está aberta?")

def main():
    print("=== Painel de Teste Independente ===")
    print("Certifique-se de que o rcssservermj já está rodando em outro terminal.")
    
    while True:
        print("\nO que você quer fazer?")
        print("1 - Posicionar Robô")
        print("2 - Posicionar e lançar a Bola")
        print("3 - Dar Play On (Voltar a rolar o jogo)")
        print("4 - Sair")
        
        escolha = input("Escolha (1/2/3/4): ")
        
        if escolha == '1':
            time = input("Time (Left/Right) [Left]: ") or "Left"
            unum = input("Número do Robô [2]: ") or "2"
            x = input("X: ") or "0.0"
            y = input("Y: ") or "0.0"
            z = input("Z: ") or "0.6"
            
            # Formato do comando S-expression do rcssservermj
            comando = f"(agent (unum {unum}) (team {time}) (pos {x} {y} {z}))"
            enviar_comando_monitor(comando)
            print(f"Comando enviado: {comando}")

        elif escolha == '2' or None:
            pos_str = input("Posição da bola (x y z) [0 0 0.2]: ") or "0 0 0.2"
            vel_str = input("Velocidade da bola (vx vy vz) [0 0 0]: ") or "0 0 0"
            
            # Pega os valores dando split nos espaços
            bx, by, bz = pos_str.split()
            vx, vy, vz = vel_str.split()
            
            # Formato do comando S-expression do rcssservermj
            comando = f"(ball (pos {bx} {by} {bz}) (vel {vx} {vy} {vz}))"
            enviar_comando_monitor(comando)
            print(f"Comando enviado: {comando}")

        elif escolha == '3':
            comando = "(playMode PlayOn)"
            enviar_comando_monitor(comando)
            print(f"Comando enviado: {comando}")

        elif escolha == '4':
            break

        else:
            pos_str = input("Posição da bola (x y z) [0 0 0.2]: ") or "0 -0.1 0.2"
            vel_str = input("Velocidade da bola (vx vy vz) [0 0 0]: ") or "0 0 0"
            
            # Pega os valores dando split nos espaços
            bx, by, bz = pos_str.split()
            vx, vy, vz = vel_str.split()

            # Formato do comando S-expression do rcssservermj
            comando = f"(ball (pos {bx} {by} {bz}) (vel {vx} {vy} {vz}))"
            enviar_comando_monitor(comando)
            print(f"Comando enviado: {comando}")

if __name__ == "__main__":
    main()
