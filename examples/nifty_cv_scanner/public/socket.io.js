(function () {
  function socketUrl(target) {
    var base = target || window.location.origin;
    var url = new URL(base, window.location.href);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    url.pathname = '/ws';
    url.search = '';
    url.hash = '';
    return url.toString();
  }

  function Socket(target) {
    this.handlers = {};
    this.target = target || window.location.origin;
    this.ws = new WebSocket(socketUrl(this.target));
    var self = this;

    this.ws.onmessage = function (event) {
      var message = JSON.parse(event.data);
      var handlers = self.handlers[message.event] || [];
      handlers.forEach(function (handler) { handler(message.data); });
    };
  }

  Socket.prototype.on = function (event, handler) {
    this.handlers[event] = this.handlers[event] || [];
    this.handlers[event].push(handler);
  };

  Socket.prototype.emit = function (event, data) {
    var payload = JSON.stringify({ event: event, data: data });
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(payload);
      return;
    }
    this.ws.addEventListener('open', function sendOnce() {
      this.removeEventListener('open', sendOnce);
      this.send(payload);
    });
  };

  window.io = {
    connect: function (target) {
      return new Socket(target);
    }
  };
}());