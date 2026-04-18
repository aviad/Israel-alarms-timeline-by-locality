export default {
  fetch(request) {
    const url = new URL(request.url);
    url.hostname = "alarms.metta.workers.dev";
    return Response.redirect(url.toString(), 301);
  },
};
