const WebSocket = require('ws')

const wss = new WebSocket.Server({ port: 8080 })

const players = {}

// ダミープレイヤーBを登録
players.B = {
  send: (message) => console.log(`ダミープレイヤーBに送信: ${message}`),
}

wss.on('connection', (ws) => {
  console.log('新しいクライアントが接続しました')

  // プレイヤーを登録
  ws.on('message', (message) => {
    const data = JSON.parse(message)
    if (data.type === 'register') {
      players[data.playerId] = ws
      console.log(`プレイヤー${data.playerId}が登録されました`)
      ws.send(
        JSON.stringify({
          type: 'message',
          content: `プレイヤー${data.playerId}が登録されました`,
        })
      )
    } else if (data.type === 'damage') {
      const targetWs = players[data.targetId]
      if (targetWs) {
        console.log(
          `プレイヤー${data.fromId}がプレイヤー${data.targetId}に${data.damage}ダメージを与えました`
        )
        targetWs.send(
          JSON.stringify({
            type: 'damage',
            from: data.fromId,
            targetId: data.targetId, // targetIdを追加
            damage: data.damage,
          })
        )
      } else {
        console.log(`ターゲットプレイヤー${data.targetId}が見つかりません`)
      }
    }
  })

  ws.on('close', () => {
    console.log('クライアントが切断されました')
    // プレイヤーリストから削除
    for (const playerId in players) {
      if (players[playerId] === ws) {
        delete players[playerId]
        console.log(`プレイヤー${playerId}が削除されました`)
        break
      }
    }
  })
})

console.log('WebSocketサーバー起動 (ポート: 8080)')
