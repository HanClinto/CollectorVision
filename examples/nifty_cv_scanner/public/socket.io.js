(function () {
  function connectionInfo(target) {
    var base = target || window.location.origin;
    var url = new URL(base, window.location.href);

    if (url.protocol === 'http:' || url.protocol === 'https:') {
      url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    }

    if (url.pathname === '/' && !url.search) {
      url.pathname = '/ws';
    }

    var socketIo = url.pathname.replace(/\/+$/, '') === '/socket.io';
    if (socketIo) {
      if (!url.searchParams.has('EIO')) url.searchParams.set('EIO', '3');
      if (!url.searchParams.has('transport')) url.searchParams.set('transport', 'websocket');
    }

    url.hash = '';
    return { mode: socketIo ? 'socket.io' : 'raw', url: url.toString() };
  }

  function Socket(target) {
    this.handlers = {};
    this.target = target || window.location.origin;
    this.queue = [];
    this.connected = false;
    this.info = connectionInfo(this.target);
    this.ws = new WebSocket(this.info.url);
    var self = this;

    this.ws.onopen = function () {
      if (self.info.mode === 'raw') {
        self.connected = true;
        self.flush();
      }
    };

    this.ws.onmessage = function (event) {
      if (self.info.mode === 'socket.io') {
        self.handleSocketIoMessage(event.data);
        return;
      }

      var message = JSON.parse(event.data);
      self.dispatch(message.event, message.data);
    };

    this.ws.onclose = function () {
      self.connected = false;
    };
  }

  Socket.prototype.dispatch = function (event, data) {
    var handlers = this.handlers[event] || [];
    handlers.forEach(function (handler) { handler(data); });
  };

  Socket.prototype.flush = function () {
    while (this.connected && this.queue.length && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(this.queue.shift());
    }
  };

  Socket.prototype.sendPacket = function (payload) {
    if (this.connected && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(payload);
      return;
    }
    this.queue.push(payload);
  };

  Socket.prototype.handleSocketIoMessage = function (data) {
    var packet = String(data);

    if (packet.charAt(0) === '0') {
      this.ws.send('40');
      return;
    }

    if (packet === '2') {
      this.ws.send('3');
      return;
    }

    if (packet.slice(0, 2) === '40') {
      this.connected = true;
      this.flush();
      return;
    }

    if (packet.slice(0, 2) === '42') {
      var message = JSON.parse(packet.slice(2));
      this.dispatch(message[0], message[1]);
    }
  };

  Socket.prototype.on = function (event, handler) {
    this.handlers[event] = this.handlers[event] || [];
    this.handlers[event].push(handler);
  };

  Socket.prototype.emit = function (event, data) {
    var payload = this.info.mode === 'socket.io'
      ? '42' + JSON.stringify([event, data])
      : JSON.stringify({ event: event, data: data });
    this.sendPacket(payload);
  };

  window.io = {
    connect: function (target) {
      return new Socket(target);
    }
  };
}());